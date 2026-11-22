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
    state = pp.preprocess(config(cascade_domain(tmp_path)))
    assert set(state.items) == {"I1", "I2", "I3", "I4", "I6"}
    assert {u for u, _ in state.pairs} == {"U2"}
    assert set(state.relations) == {"author"}           # sequel lost its only triple in round 1
    assert state.stats["filter_rounds"] == 2
    assert state.stats["isolated_items_removed"] == 1   # I5
    assert state.stats["items_removed"] == 2            # I9 (round 1), I7 (round 2)
    assert state.stats["users_removed"] == 1            # U1 (round 2)
    assert state.stats["pairs_removed"] == 5            # (U1, I5) + U1's remaining 4


def test_result_is_a_fixed_point(tmp_path):
    state = pp.preprocess(config(cascade_domain(tmp_path)))
    assert pp.filter_users(state, 5) == 0
    assert pp.filter_items(state, 1) == 0
    assert pp.remove_isolated_items(state) == 0


def test_item_only_as_tail_is_not_isolated(tmp_path):
    state = merged(tmp_path, items__csv=VALID["items.csv"] + "I4,Book Four\n",
                   kg__csv=VALID["kg.csv"] + "item,I3,sequel,item,I4\n")
    assert pp.remove_isolated_items(state) == 0
    assert "I4" in state.items


# ---------------------------------------------------------------- real data

def domain_or_skip(name):
    domain = ROOT / "data" / name
    if not (domain / "raw" / "interactions.jsonl").exists():
        pytest.skip(f"{name} not generated")
    return domain


def structure(state):
    return set(state.pairs), state.items, state.auxs, state.relations, state.kg


def test_noisy_book_cleans_to_book():
    book = pp.preprocess(config(domain_or_skip("book")))
    noisy = pp.preprocess(config(domain_or_skip("book_noisy")))
    assert (len(book.pairs), len(book.items), len(book.auxs), len(book.relations), len(book.kg)) == (
        169_273, 8_475, 23_330, 20, 65_190)
    for name, b, n in zip(("pairs", "items", "auxs", "relations", "kg"), structure(book), structure(noisy)):
        assert b == n, name
    assert noisy.stats["users_removed"] == 400 and noisy.stats["items_removed"] == 271
    assert book.stats["filter_rounds"] == noisy.stats["filter_rounds"] == 1
    assert "isolated_items_removed" not in noisy.stats
