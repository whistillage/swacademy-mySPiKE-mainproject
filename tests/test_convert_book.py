import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import convert_book as cb  # noqa: E402

AUTHOR_BOOK = Path(__file__).resolve().parents[2] / "SPiKE/spike/datasets/kg/amazon-book_kg"


# ---------------------------------------------------------------- helpers

def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write(path, text, newline="\n"):
    path.write_text(text.replace("\n", newline), encoding="utf-8", newline="")


@pytest.fixture
def tiny_author(tmp_path):
    """Minimal author-format folder: 2 items, 3 aux, 2 users, CRLF review file."""
    src = tmp_path / "src"
    src.mkdir()
    write(src / "user_list.txt", "org_id remap_id\nU_A 0\nU_B 1\n")
    write(src / "item_list.txt",
          "ori_id remap_id dbpedia_id\n000100039X 0 res:The_Prophet_(book)\n0002005395 1 res:Deafening_(novel)\n")
    write(src / "entity_list.txt",
          "ori_id remap_id\nThe_Prophet_(book) 0\nDeafening_(novel) 1\n"
          "cat:Philosophical_fiction 2\nres:Kahlil_Gibran 3\nhttp://example.com 4\n")
    write(src / "relation_list.txt", "ori_id remap_id\ndct:subject 0\ndbo:author 1\n")
    write(src / "interaction_list.txt", "user_id item_id\n0 0\n1 0\n1 1\n")
    write(src / "review_list.txt",
          'user_id item_id review\n0 0 Great, "really" great.\n1 0 \n1 1 Two  spaces kept\n', newline="\r\n")
    write(src / "kg_final.txt", "0 0 2\n0 1 3\n0 1 3\n1 1 4\n")
    return src


# ---------------------------------------------------------------- unit tests

@pytest.mark.parametrize("entity_id, name", [
    ("res:The_Prophet_(book)", "The Prophet (book)"),
    ("cat:Alfred_A._Knopf_books", "Alfred A. Knopf books"),
    ("http://www.monideepa.blogspot.com", "http://www.monideepa.blogspot.com"),
])
def test_readable_name(entity_id, name):
    assert cb.readable_name(entity_id) == name


@pytest.mark.parametrize("aux_id, aux_type", [
    ("cat:Fables", "category"),
    ("res:Kahlil_Gibran", "resource"),
    ("http://www.palinstravels.co.uk/", "url"),
])
def test_aux_type_of(aux_id, aux_type):
    assert cb.aux_type_of(aux_id) == aux_type


def test_aux_type_unknown_prefix():
    with pytest.raises(ValueError):
        cb.aux_type_of("dbo:Something")


def test_templates_follow_spec_rules():
    assert len(cb.RELATION_TEMPLATES) == 20
    for sentence, reverse in cb.RELATION_TEMPLATES.values():
        assert "{head}" in sentence and "{tails}" in sentence
        assert "{tail}" in reverse and "{head}" in reverse
        for t in (sentence, reverse):
            assert "'" not in t and '"' not in t and not t.endswith(".")


def test_convert_tiny(tiny_author, tmp_path):
    out = tmp_path / "raw"
    stats = cb.convert(tiny_author, out)
    assert stats == {"interactions": 3, "empty_reviews_filled": 1, "items": 2,
                     "auxs": 3, "kg_rows": 4, "relations": 2}

    inter = read_jsonl(out / "interactions.jsonl")
    assert inter == [
        {"user_id": "U_A", "item_id": "000100039X", "review": 'Great, "really" great.'},
        {"user_id": "U_B", "item_id": "000100039X", "review": cb.EMPTY_REVIEW_PLACEHOLDER},
        {"user_id": "U_B", "item_id": "0002005395", "review": "Two  spaces kept"},
    ]
    assert read_csv(out / "items.csv") == [
        {"item_id": "000100039X", "name": "The Prophet (book)"},
        {"item_id": "0002005395", "name": "Deafening (novel)"},
    ]
    assert read_csv(out / "auxs.csv") == [
        {"aux_id": "cat:Philosophical_fiction", "name": "Philosophical fiction", "aux_type": "category"},
        {"aux_id": "res:Kahlil_Gibran", "name": "Kahlil Gibran", "aux_type": "resource"},
        {"aux_id": "http://example.com", "name": "http://example.com", "aux_type": "url"},
    ]
    kg = read_csv(out / "kg.csv")
    assert len(kg) == 4  # duplicate row kept
    assert kg[0] == {"head_type": "item", "head": "000100039X", "relation": "dct:subject",
                     "tail_type": "aux", "tail": "cat:Philosophical_fiction"}
    rel = read_csv(out / "relations.csv")
    assert [r["relation"] for r in rel] == ["dct:subject", "dbo:author"]
    assert rel[1]["reverse_sentence"] == "{tail} is credited as the author of {head}"


def test_leading_zero_ids_survive(tiny_author, tmp_path):
    out = tmp_path / "raw"
    cb.convert(tiny_author, out)
    text = (out / "items.csv").read_text(encoding="utf-8")
    assert "\n0002005395," in text


def test_main_refuses_overwrite(tiny_author, tmp_path):
    out = tmp_path / "raw"
    cb.main([str(tiny_author), str(out)])
    with pytest.raises(SystemExit):
        cb.main([str(tiny_author), str(out)])
    cb.main([str(tiny_author), str(out), "--force"])


# ---------------------------------------------------------------- real author data

@pytest.mark.skipif(not AUTHOR_BOOK.exists(), reason="author book data not available")
def test_real_book_counts_and_round_trip(tmp_path):
    out = tmp_path / "raw"
    stats = cb.convert(AUTHOR_BOOK, out)
    assert stats == {"interactions": 169_273, "empty_reviews_filled": 5, "items": 8_696,
                     "auxs": 23_867, "kg_rows": 69_573, "relations": 20}

    # Map every original ID back to the author's remap ID and compare with the source files.
    users = {v: k for k, v in cb.read_id_map(AUTHOR_BOOK / "user_list.txt").items()}
    entities = {v: k for k, v in cb.read_id_map(AUTHOR_BOOK / "entity_list.txt").items()}
    relations = {v: k for k, v in cb.read_id_map(AUTHOR_BOOK / "relation_list.txt").items()}
    item_by_asin = {asin: k for k, (asin, _) in cb.read_items(AUTHOR_BOOK / "item_list.txt").items()}

    kg_back = [(item_by_asin[r["head"]], relations[r["relation"]], entities[r["tail"]])
               for r in read_csv(out / "kg.csv")]
    assert kg_back == list(cb.read_kg(AUTHOR_BOOK / "kg_final.txt"))

    inter = read_jsonl(out / "interactions.jsonl")
    pairs_back = [(users[r["user_id"]], item_by_asin[r["item_id"]]) for r in inter]
    assert pairs_back == list(cb.read_pairs(AUTHOR_BOOK / "interaction_list.txt"))

    src_reviews = [t for _, _, t in cb.read_reviews(AUTHOR_BOOK / "review_list.txt")]
    for row, src in zip(inter, src_reviews):
        assert row["review"] == (src if src.strip() else cb.EMPTY_REVIEW_PLACEHOLDER)

    aux_ids = [r["aux_id"] for r in read_csv(out / "auxs.csv")]
    assert aux_ids == [entities_id for entities_id, idx in sorted(entities.items(), key=lambda x: x[1])
                       if idx >= 8_696]
