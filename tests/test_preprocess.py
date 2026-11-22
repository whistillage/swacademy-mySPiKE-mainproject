import csv
import hashlib
import json
import os
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


def merged(tmp_path, **changes) -> pp.State:
    """State after [1] validation and [2] merge only."""
    return pp.merge_duplicates(pp.run_validation(make_domain(tmp_path, **changes) / "raw"))


def row(user, item, review="r", **extra):
    return {"user_id": user, "item_id": item, "review": review, **extra}


# ---------------------------------------------------------------- config / CLI

@pytest.mark.parametrize("split", [(0.7, 0.2, 0.2), (0.0, 0.5, 0.5), (0.5, 0.5)])
def test_bad_split_is_rejected(tmp_path, split):
    with pytest.raises(ValueError):
        config(tmp_path, split=split).check()


@pytest.mark.parametrize("user_min", [0, 1, 2])
def test_user_min_below_three_is_rejected(tmp_path, user_min):
    with pytest.raises(ValueError):
        config(tmp_path, user_min_interactions=user_min).check()


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
    pp.preprocess(config(make_domain(tmp_path)), write=False)
    out = capsys.readouterr().out
    assert "[1] 검증 통과: 경고" in out and "== WARNING" not in out


# ---------------------------------------------------------------- [2] merge duplicates

def test_merge_keeps_reviews_in_input_order(tmp_path):
    rows = VALID["interactions.jsonl"] + [row("U1", "I1", "second"), row(" U1 ", "I1", "third")]
    state = merged(tmp_path, interactions__jsonl=rows)
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
    state = merged(tmp_path, interactions__jsonl=rows)
    assert state.optional_keys == ("timestamp", "rating")
    assert state.pairs[("U1", "I1")]["reviews"] == [
        {"text": "a", "timestamp": 1167609600, "rating": 5.0},
        {"text": "b", "timestamp": None, "rating": 4}]


def test_merge_without_optional_columns_has_text_only(tmp_path):
    state = merged(tmp_path)
    assert all(set(r) == {"text"} for e in state.pairs.values() for r in e["reviews"])


def test_kg_duplicates_collapse(tmp_path):
    state = merged(tmp_path, kg__csv=VALID["kg.csv"] + "item,I1,author,aux,A1\n" * 2)
    assert len(state.kg) == 4
    assert state.stats["kg_duplicate_rows_removed"] == 2


def test_lists_are_loaded(tmp_path):
    state = merged(tmp_path)
    assert state.items == {"I1": "Book One", "I2": "Book Two", "I3": "Book Three"}
    assert state.auxs["A1"] == ("Author One", "person")
    assert state.relations["sequel"]["tail_type"] == "item"


# ---------------------------------------------------------------- [3] user filter

def test_user_filter_counts_distinct_items(tmp_path):
    rows = [row("U1", i) for i in ("I1", "I2", "I3")] + [row("U2", "I1")] * 3 + [row("U2", "I2")]
    state = merged(tmp_path, interactions__jsonl=rows)
    assert pp.filter_users(state, 3) == 1            # U2: 4 rows but 2 distinct items
    assert {u for u, _ in state.pairs} == {"U1"}
    assert state.stats["users_removed"] == 1 and state.stats["pairs_removed"] == 2


# ---------------------------------------------------------------- [4] item filter + KG cleanup

REL_EXTRA = ("genre,item,aux,{head} is {tails},{tail} covers {head}\n"
             "related,aux,aux,{head} is related to {tails},{tail} is related to {head}\n")


def kg_state(tmp_path):
    """I4 has no interactions. Its triples: I4 -author-> A1 (A1 also used by I1, I2),
    I4 -genre-> A3 (A3 and genre used only by I4), I2 -sequel-> I4.
    A3 -related-> A1 is the only aux -> aux triple (related used nowhere else)."""
    return merged(
        tmp_path,
        items__csv=VALID["items.csv"] + "I4,Book Four\n",
        auxs__csv=VALID["auxs.csv"] + "A3,Genre Three,genre\n",
        relations__csv=VALID["relations.csv"] + REL_EXTRA,
        kg__csv=VALID["kg.csv"] + "item,I4,author,aux,A1\nitem,I4,genre,aux,A3\n"
                                  "aux,A3,related,aux,A1\nitem,I2,sequel,item,I4\n")


def test_item_filter_and_kg_cleanup(tmp_path):
    state = kg_state(tmp_path)
    assert pp.filter_items(state, 1) == 1
    assert "I4" not in state.items
    # step 1: every triple with I4 is gone (as head and as tail)
    assert not any("I4" in (t[1], t[4]) for t in state.kg)
    # step 2: A3 lost its only item; A1 is still used by I1 and I2
    assert "A3" not in state.auxs and "A1" in state.auxs
    # step 3: the aux -> aux triple with A3 is gone
    assert not any(t[0] == "aux" for t in state.kg)
    # step 4: genre and related have no triples left; author and sequel stay
    assert set(state.relations) == {"author", "sequel"}
    assert state.kg == {("item", "I1", "author", "aux", "A1"), ("item", "I2", "author", "aux", "A1"),
                        ("item", "I3", "author", "aux", "A2"), ("item", "I1", "sequel", "item", "I2")}
    assert {k: state.stats[k] for k in ("items_removed", "aux_removed", "kg_triples_removed",
                                        "relations_removed")} == {
        "items_removed": 1, "aux_removed": 1, "kg_triples_removed": 4, "relations_removed": 2}


def test_item_filter_keeps_everything_when_all_items_are_used(tmp_path):
    state = merged(tmp_path)
    kg_before = set(state.kg)
    assert pp.filter_items(state, 1) == 0
    assert state.kg == kg_before and "items_removed" not in state.stats


def test_item_min_above_one_also_removes_pairs(tmp_path):
    state = merged(tmp_path)                        # I1, I2, I3 have one user each
    pp.filter_items(state, 2)
    assert state.items == {} and state.pairs == {} and state.kg == set()
    assert state.stats["pairs_removed"] == 3


# ---------------------------------------------------------------- [5] isolated items + repeat

def cascade_domain(tmp_path):
    """Round 1: I9 has no users -> removed with I5 -sequel-> I9 -> I5 is isolated -> removed
    with (U1, I5) -> U1 drops to 4 items.  Round 2: U1 removed -> I7 (only U1) removed.
    No interaction is removed in round 2's [4]/[5] -> stop after 2 rounds."""
    items = "item_id,name\n" + "".join(f"I{k},Book {k}\n" for k in (1, 2, 3, 4, 5, 6, 7, 9))
    kg = ("head_type,head,relation,tail_type,tail\n"
          + "".join(f"item,I{k},author,aux,A1\n" for k in (1, 2, 3, 4, 6, 7))
          + "item,I5,sequel,item,I9\n")
    rows = [row("U1", f"I{k}") for k in (1, 2, 3, 5, 7)] + [row("U2", f"I{k}") for k in (1, 2, 3, 4, 6)]
    return make_domain(tmp_path, items__csv=items, auxs__csv="aux_id,name,aux_type\nA1,Author One,person\n",
                       kg__csv=kg, interactions__jsonl=rows)


def test_filters_repeat_until_nothing_changes(tmp_path):
    state = pp.preprocess(config(cascade_domain(tmp_path)), write=False)
    assert set(state.items) == {"I1", "I2", "I3", "I4", "I6"}
    assert {u for u, _ in state.pairs} == {"U2"}
    assert set(state.relations) == {"author"}           # sequel lost its only triple in round 1
    assert state.stats["filter_rounds"] == 2
    assert state.stats["isolated_items_removed"] == 1   # I5
    assert state.stats["items_removed"] == 2            # I9 (round 1), I7 (round 2)
    assert state.stats["users_removed"] == 1            # U1 (round 2)
    assert state.stats["pairs_removed"] == 5            # (U1, I5) + U1's remaining 4


def test_result_is_a_fixed_point(tmp_path):
    state = pp.preprocess(config(cascade_domain(tmp_path)), write=False)
    assert pp.filter_users(state, 5) == 0
    assert pp.filter_items(state, 1) == 0
    assert pp.remove_isolated_items(state) == 0


def test_item_only_as_tail_is_not_isolated(tmp_path):
    state = merged(tmp_path, items__csv=VALID["items.csv"] + "I4,Book Four\n",
                   kg__csv=VALID["kg.csv"] + "item,I3,sequel,item,I4\n")
    assert pp.remove_isolated_items(state) == 0
    assert "I4" in state.items


# ---------------------------------------------------------------- [6] split

@pytest.mark.parametrize("n, expected", [
    (3, (1, 1, 1)), (5, (3, 1, 1)), (6, (4, 1, 1)), (7, (5, 1, 1)), (10, (7, 1, 2)), (20, (14, 2, 4)),
    (15, (10, 2, 3)),     # valid: 0.1 * 15 = 1.5 -> 2 (half up), test: 3
    (25, (17, 3, 5)),     # valid: 2.5 -> 3 (round() would give 2)
])
def test_split_counts(n, expected):
    assert pp.split_counts(n, (0.7, 0.1, 0.2)) == expected


def test_split_counts_without_train_item_is_an_error():
    with pytest.raises(ValueError):
        pp.split_counts(3, (0.1, 0.1, 0.8))


def test_split_user_assigns_every_item_once():
    items = [f"I{k:02d}" for k in range(20)]
    s = pp.split_user("U1", items, 2020, (0.7, 0.1, 0.2))
    assert (len(s["train"]), len(s["valid"]), len(s["test"])) == (14, 2, 4)
    assert sorted(s["train"] + s["valid"] + s["test"]) == items


def test_split_user_ignores_item_order_and_other_users():
    items = [f"I{k:02d}" for k in range(10)]
    a = pp.split_user("U1", items, 2020, (0.7, 0.1, 0.2))
    assert pp.split_user("U1", list(reversed(items)), 2020, (0.7, 0.1, 0.2)) == a
    assert pp.split_user("U2", items, 2020, (0.7, 0.1, 0.2)) != a       # each user has its own shuffle
    assert pp.split_user("U1", items, 7, (0.7, 0.1, 0.2)) != a          # and the seed matters


def test_split_users_is_independent_of_other_users(tmp_path):
    state = merged(tmp_path)
    state.pairs = {(u, f"I{k}"): {} for u in ("U1", "U2") for k in range(1, 8)}
    pp.split_users(state, 2020, (0.7, 0.1, 0.2))
    with_u2 = state.split["U1"]
    state.pairs = {k: v for k, v in state.pairs.items() if k[0] == "U1"}
    pp.split_users(state, 2020, (0.7, 0.1, 0.2))
    assert state.split["U1"] == with_u2
    assert state.stats["split_pairs"] == {"train": 5, "valid": 1, "test": 1}


# ---------------------------------------------------------------- [7] IDs

def test_numbering_uses_code_point_order():
    assert pp.numbering({"2", "10", "b", "B", "가", "_x"}) == {"10": 0, "2": 1, "B": 2, "_x": 3, "b": 4, "가": 5}


def test_assign_ids(tmp_path):
    state = merged(tmp_path)
    ids = pp.assign_ids(state)
    assert ids.user == {"U1": 0, "U2": 1}
    assert ids.item == {"I1": 0, "I2": 1, "I3": 2}
    assert ids.aux == {"A1": 0, "A2": 1}
    assert ids.relation == {"author": 0, "sequel": 1}
    assert [ids.node("item", i) for i in ("I1", "I2", "I3")] == [0, 1, 2]
    assert [ids.node("aux", a) for a in ("A1", "A2")] == [3, 4]          # aux come after the items


def test_ids_do_not_depend_on_input_order(tmp_path):
    a = merged(tmp_path / "a")
    rev = lambda text: text.split("\n", 1)[0] + "\n" + "\n".join(reversed(text.split("\n", 1)[1].strip().split("\n"))) + "\n"
    b = merged(tmp_path / "b", items__csv=rev(VALID["items.csv"]), auxs__csv=rev(VALID["auxs.csv"]),
               relations__csv=rev(VALID["relations.csv"]), kg__csv=rev(VALID["kg.csv"]),
               interactions__jsonl=list(reversed(VALID["interactions.jsonl"])))
    assert pp.assign_ids(a) == pp.assign_ids(b)


# ---------------------------------------------------------------- [8] write

def small_domain(tmp_path):
    """3 users with >= 3 items each (run with user_min 3). I5 is used by nobody -> removed.
    IDs are given out of sorted order on purpose."""
    items = "item_id,name\nI3,Book Three\nI1,Book One\nI4,Book Four\nI2,Book Two\nI5,Unused\n"
    auxs = "aux_id,name,aux_type\nA2,Genre Two,genre\nA1,Author One,person\nA9,Only Unused,person\n"
    kg = ("head_type,head,relation,tail_type,tail\n"
          "item,I1,author,aux,A1\nitem,I2,author,aux,A1\nitem,I3,author,aux,A2\nitem,I4,author,aux,A2\n"
          "item,I5,author,aux,A9\nitem,I1,sequel,item,I2\nitem,I1,author,aux,A1\n")
    rows = ([row("U2", i, f"u2 {i}") for i in ("I1", "I2", "I3")]
            + [row("U1", i, f"u1 {i}") for i in ("I4", "I3", "I2", "I1")]
            + [row("U3", i, f"u3 {i}") for i in ("I2", "I3", "I4")]
            + [row("U1", "I1", "u1 I1 again")])
    return make_domain(tmp_path, items__csv=items, auxs__csv=auxs, kg__csv=kg, interactions__jsonl=rows)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


@pytest.fixture
def written(tmp_path):
    domain = small_domain(tmp_path)
    state = pp.preprocess(config(domain, user_min_interactions=3))
    return domain / "prep" / "v1", state


def test_writes_the_ten_files(written):
    out, _ = written
    assert sorted(p.name for p in out.iterdir()) == sorted(pp.OUTPUT_FILES)
    assert not any(p.name.startswith(".") for p in out.parent.iterdir())     # temporary folder renamed


def test_lists(written):
    out, _ = written
    assert read_csv(out / "user_list.csv") == [["user_id", "user_idx"], ["U1", "0"], ["U2", "1"], ["U3", "2"]]
    assert read_csv(out / "item_list.csv") == [
        ["item_id", "item_idx", "node_idx", "name"],
        ["I1", "0", "0", "Book One"], ["I2", "1", "1", "Book Two"],
        ["I3", "2", "2", "Book Three"], ["I4", "3", "3", "Book Four"]]
    assert read_csv(out / "aux_list.csv") == [
        ["aux_id", "aux_idx", "node_idx", "name", "aux_type"],
        ["A1", "0", "4", "Author One", "person"], ["A2", "1", "5", "Genre Two", "genre"]]
    assert read_csv(out / "relation_list.csv")[1:] == [
        ["author", "0", "item", "aux", "{head} lists {tails} as its author", "{tail} is credited as the author of {head}"],
        ["sequel", "1", "item", "item", "{head} is followed by {tails}", "{tail} is the sequel to {head}"]]


def test_kg_uses_node_and_relation_numbers(written):
    out, _ = written
    assert read_csv(out / "kg.csv") == [
        ["head", "relation", "tail"],
        ["0", "0", "4"], ["0", "1", "1"], ["1", "0", "4"], ["2", "0", "5"], ["3", "0", "5"]]


def test_split_files(written):
    out, state = written
    lines = {part: (out / f"{part}.txt").read_text(encoding="utf-8").splitlines() for part in ("train", "valid", "test")}
    users = {"U1": {0, 1, 2, 3}, "U2": {0, 1, 2}, "U3": {1, 2, 3}}
    for uidx, (u, items) in enumerate(sorted(users.items())):
        got = []
        for part in ("train", "valid", "test"):
            fields = list(map(int, lines[part][uidx].split()))
            assert fields[0] == uidx and fields[1:] == sorted(fields[1:]) and len(fields) > 1
            got += fields[1:]
        assert sorted(got) == sorted(items)                              # every item exactly once
    assert [len(lines[p]) for p in lines] == [3, 3, 3]


def test_interactions_file(written):
    out, _ = written
    rows = [json.loads(line) for line in (out / "interactions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["user_idx"], r["item_idx"]) for r in rows] == [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]
    assert rows[0] == {"user_idx": 0, "item_idx": 0, "count": 2,
                       "reviews": [{"text": "u1 I1"}, {"text": "u1 I1 again"}]}


def test_meta(written):
    out, _ = written
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert list(meta) == ["domain", "version", "n_users", "n_items", "n_aux", "n_relations", "node_ranges",
                          "config", "stats", "input_hash"]
    assert (meta["domain"], meta["version"], meta["n_users"], meta["n_items"], meta["n_aux"], meta["n_relations"]) == (
        "dom", "v1", 3, 4, 2, 2)
    assert meta["node_ranges"] == {"item": [0, 3], "aux": [4, 5]}
    assert meta["config"] == {"seed": 2020, "split": [0.7, 0.1, 0.2], "rounding": "half_up",
                              "user_min_interactions": 3, "item_min_interactions": 1}
    assert list(meta["stats"]) == list(pp.STATS_KEYS)                    # every key, fixed order
    assert meta["stats"]["isolated_items_removed"] == 0                  # zero, not missing
    assert (meta["stats"]["items_removed"], meta["stats"]["aux_removed"], meta["stats"]["duplicate_rows_merged"],
            meta["stats"]["kg_duplicate_rows_removed"]) == (1, 1, 1, 1)
    raw = out.parent.parent / "raw"
    assert meta["input_hash"] == {n: "sha256:" + hashlib.sha256((raw / n).read_bytes()).hexdigest()
                                  for n in ("interactions.jsonl", "items.csv", "auxs.csv", "kg.csv", "relations.csv")}


def test_existing_version_is_not_overwritten(written, capsys):
    out, _ = written
    before = (out / "meta.json").read_bytes()
    assert pp.main([str(out.parent.parent), "--version", "v1", "--user-min", "3"]) == 3
    assert "[1] 검증" not in capsys.readouterr().out                     # stopped before any work
    assert (out / "meta.json").read_bytes() == before


def test_leftover_temporary_folder_is_replaced(tmp_path):
    domain = small_domain(tmp_path)
    leftover = domain / "prep" / ".v1.tmp"
    leftover.mkdir(parents=True)
    (leftover / "junk.txt").write_text("x")
    pp.preprocess(config(domain, user_min_interactions=3))
    assert not leftover.exists() and not (domain / "prep" / "v1" / "junk.txt").exists()


# ---------------------------------------------------------------- real data

def domain_or_skip(name):
    domain = ROOT / "data" / name
    if not (domain / "raw" / "interactions.jsonl").exists():
        pytest.skip(f"{name} not generated")
    return domain


def structure(state):
    return set(state.pairs), state.items, state.auxs, state.relations, state.kg


def test_noisy_book_cleans_to_book():
    book = pp.preprocess(config(domain_or_skip("book")), write=False)
    noisy = pp.preprocess(config(domain_or_skip("book_noisy")), write=False)
    assert (len(book.pairs), len(book.items), len(book.auxs), len(book.relations), len(book.kg)) == (
        169_273, 8_475, 23_330, 20, 65_190)
    for name, b, n in zip(("pairs", "items", "auxs", "relations", "kg"), structure(book), structure(noisy)):
        assert b == n, name
    assert noisy.stats["users_removed"] == 400 and noisy.stats["items_removed"] == 271
    assert book.stats["filter_rounds"] == noisy.stats["filter_rounds"] == 1
    assert "isolated_items_removed" not in noisy.stats
    assert book.split == noisy.split
    assert book.ids == noisy.ids
    assert (len(book.ids.user), book.ids.n_items, len(book.ids.aux), len(book.ids.relation)) == (14_946, 8_475, 23_330, 20)
    assert book.ids.node("aux", max(book.ids.aux, key=book.ids.aux.get)) == 31_804
    assert book.stats["split_pairs"] == {"train": 115_376, "valid": 20_686, "test": 33_211}   # 68.2 / 12.2 / 19.6


def test_real_book_write(tmp_path):
    """Run on the real raw files (linked, so nothing is written under data/)."""
    raw = domain_or_skip("book") / "raw"
    domain = tmp_path / "book"
    domain.mkdir()
    os.symlink(raw.resolve(), domain / "raw")
    pp.preprocess(config(domain))
    out = domain / "prep" / "v1"
    assert [len((out / f"{p}.txt").read_text().splitlines()) for p in ("train", "valid", "test")] == [14_946] * 3
    assert len(read_csv(out / "kg.csv")) == 65_190 + 1
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["node_ranges"] == {"item": [0, 8474], "aux": [8475, 31804]}
    assert meta["stats"]["split_pairs"] == {"train": 115_376, "valid": 20_686, "test": 33_211}
