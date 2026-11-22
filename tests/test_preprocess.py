from pathlib import Path

import pytest

import preprocess as pp
from dataset_fixtures import VALID, write_raw

ROOT = Path(__file__).resolve().parent.parent


def make_domain(tmp_path, **changes):
    files = dict(VALID)
    for key, value in changes.items():
        files[key.replace("__", ".")] = value
    write_raw(tmp_path / "dom" / "raw", files)
    return tmp_path / "dom"


def config(domain, **kw):
    return pp.Config(domain_dir=domain, version="v1", **kw)


def row(user, item, review="r", **extra):
    return {"user_id": user, "item_id": item, "review": review, **extra}


# ---------------------------------------------------------------- config / CLI

@pytest.mark.parametrize("split", [(0.7, 0.2, 0.2), (0.0, 0.5, 0.5), (0.5, 0.5)])
def test_bad_split_is_rejected(tmp_path, split):
    with pytest.raises(ValueError):
        config(tmp_path, split=split).check()


def test_cli_parses_options():
    c = pp.parse_args(["data/x", "--version", "v2", "--seed", "7", "--split", "0.8", "0.1", "0.1",
                       "--user-min", "3", "--item-min", "2"])
    assert c == pp.Config(Path("data/x"), "v2", 7, (0.8, 0.1, 0.1), "half_up", 3, 2)
    assert c.raw_dir == Path("data/x/raw") and c.out_dir == Path("data/x/prep/v2")


# ---------------------------------------------------------------- [1] validate

def test_validation_error_stops(tmp_path, capsys):
    domain = make_domain(tmp_path, items__csv=VALID["items.csv"] + "I2,Again\n")
    assert pp.main([str(domain), "--version", "v1"]) == 1
    out = capsys.readouterr().out
    assert "E_DUP_ITEM" in out and "전처리를 중단합니다" in out


def test_validation_pass_prints_one_line(tmp_path, capsys):
    pp.preprocess(config(make_domain(tmp_path)))
    out = capsys.readouterr().out
    assert "[1] 검증 통과: 경고" in out and "== WARNING" not in out


# ---------------------------------------------------------------- [2] merge duplicates

def test_merge_keeps_reviews_in_input_order(tmp_path):
    rows = VALID["interactions.jsonl"] + [row("U1", "I1", "second"), row(" U1 ", "I1", "third")]
    state = pp.preprocess(config(make_domain(tmp_path, interactions__jsonl=rows)))
    assert state.pairs[("U1", "I1")] == {
        "count": 3, "reviews": [{"text": "Great book."}, {"text": "second"}, {"text": "third"}]}
    assert len(state.pairs) == 3
    assert state.stats["duplicate_pairs_merged"] == 1
    assert state.stats["duplicate_rows_merged"] == 2


def test_merge_carries_optional_columns(tmp_path):
    rows = [row("U1", "I1", "a", timestamp=1167609600, rating=5.0),
            row("U1", "I1", "b", timestamp=None, rating=4),
            row("U1", "I2", "c", timestamp=1, rating=None),
            row("U2", "I3", "d", timestamp=2, rating=3.5)]
    state = pp.preprocess(config(make_domain(tmp_path, interactions__jsonl=rows)))
    assert state.optional_keys == ("timestamp", "rating")
    assert state.pairs[("U1", "I1")]["reviews"] == [
        {"text": "a", "timestamp": 1167609600, "rating": 5.0},
        {"text": "b", "timestamp": None, "rating": 4}]


def test_merge_without_optional_columns_has_text_only(tmp_path):
    state = pp.preprocess(config(make_domain(tmp_path)))
    assert all(set(r) == {"text"} for e in state.pairs.values() for r in e["reviews"])


def test_kg_duplicates_collapse(tmp_path):
    state = pp.preprocess(config(make_domain(tmp_path, kg__csv=VALID["kg.csv"] + "item,I1,author,aux,A1\n" * 2)))
    assert len(state.kg) == 4
    assert state.stats["kg_duplicate_rows_removed"] == 2


def test_lists_are_loaded(tmp_path):
    state = pp.preprocess(config(make_domain(tmp_path)))
    assert state.items == {"I1": "Book One", "I2": "Book Two", "I3": "Book Three"}
    assert state.auxs["A1"] == ("Author One", "person")
    assert state.relations["sequel"]["tail_type"] == "item"


# ---------------------------------------------------------------- real data

def test_noisy_book_merge():
    domain = ROOT / "data" / "book_noisy"
    if not (domain / "raw" / "interactions.jsonl").exists():
        pytest.skip("book_noisy not generated")
    state = pp.preprocess(config(domain))
    assert len(state.pairs) == 170_407
    assert state.stats == {"duplicate_pairs_merged": 600, "duplicate_rows_merged": 650,
                           "kg_duplicate_rows_removed": 3_068}
    assert len(state.kg) == 67_014
