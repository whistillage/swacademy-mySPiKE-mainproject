import collections
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_noisy_book as mn  # noqa: E402

CLEAN_RAW = Path(__file__).resolve().parent.parent / "data/book/raw"

pytestmark = pytest.mark.skipif(not (CLEAN_RAW / "interactions.jsonl").exists(),
                                reason="data/book/raw not generated (run tools/convert_book.py)")


def pairs_of(path):
    with open(path, encoding="utf-8") as f:
        return [(r["user_id"], r["item_id"], r["review"]) for r in map(json.loads, f)]


@pytest.fixture(scope="module")
def noisy_all(tmp_path_factory):
    out = tmp_path_factory.mktemp("book_noisy")
    summary = mn.make_noisy(CLEAN_RAW, out, seed=2020)
    return out, summary


@pytest.fixture(scope="module")
def noisy(noisy_all):
    out, summary = noisy_all
    return out, summary["interactions.jsonl"]


def test_injections_are_present(noisy):
    out, s = noisy
    clean = pairs_of(CLEAN_RAW / "interactions.jsonl")
    rows = pairs_of(out / "raw/interactions.jsonl")
    assert rows[:len(clean)] == clean                      # original rows untouched, in order
    assert len(rows) == s["noisy_rows"] == len(clean) + s["extra_rows"]
    assert all(r.strip() for _, _, r in rows)              # spec: no empty review

    pair_count = collections.Counter((u, i) for u, i, _ in rows)
    repeated = {p: c for p, c in pair_count.items() if c > 1}
    # 500 injected duplicate pairs + 100 trap users with one repeated row each
    assert len(repeated) == s["duplicate_pairs"] + s["new_users_duplicate_trap"]
    assert sorted(collections.Counter(repeated.values()).items()) == [(2, 450 + 100), (3, 50)]

    distinct = collections.Counter(u for u, _ in pair_count)
    small = {u for u, n in distinct.items() if n < 5}
    assert len(small) == s["new_users_small"] + s["new_users_duplicate_trap"]
    assert all(u.startswith(mn.NEW_USER_PREFIX) for u in small)
    # trap users have 5 rows but only 4 distinct items
    rows_per_user = collections.Counter(u for u, _, _ in rows)
    traps = [u for u in small if rows_per_user[u] == 5]
    assert len(traps) == s["new_users_duplicate_trap"]


def test_cleaning_recovers_clean_pairs(noisy):
    """Merge duplicates -> drop users < 5 -> must give back exactly the clean pair set."""
    out, _ = noisy
    clean = {(u, i) for u, i, _ in pairs_of(CLEAN_RAW / "interactions.jsonl")}
    merged = {(u, i) for u, i, _ in pairs_of(out / "raw/interactions.jsonl")}
    per_user = collections.Counter(u for u, _ in merged)
    kept = {(u, i) for u, i in merged if per_user[u] >= 5}
    assert kept == clean


def test_relations_phantom_appended(noisy_all):
    out, summary = noisy_all
    clean = (CLEAN_RAW / "relations.csv").read_bytes()
    noisy = (out / "raw/relations.csv").read_bytes()
    assert noisy.startswith(clean)                         # original rows byte-identical

    with open(out / "raw/relations.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    new = rows[20:]
    assert summary["relations.csv"] == {"clean_rows": 20, "noisy_rows": 21, "phantom_relations": 1}
    assert [r["relation"] for r in new] == ["noisy:phantomLink"]
    assert len({r["relation"] for r in rows}) == len(rows)  # spec: no duplicate relation
    for r in new:                                          # spec 6.5 rules still hold
        assert (r["head_type"], r["tail_type"]) == ("item", "aux")
        assert "{head}" in r["sentence"] and "{tails}" in r["sentence"]
        assert "{tail}" in r["reverse_sentence"] and "{head}" in r["reverse_sentence"]
        for t in (r["sentence"], r["reverse_sentence"]):
            assert "'" not in t and '"' not in t and not t.endswith(".")


def test_items_phantom_appended(noisy_all):
    out, summary = noisy_all
    assert (out / "raw/items.csv").read_bytes().startswith((CLEAN_RAW / "items.csv").read_bytes())

    with open(out / "raw/items.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    new = rows[8_696:]
    assert summary["items.csv"] == {"clean_rows": 8_696, "noisy_rows": 8_746, "phantom_items": 50}
    assert all(r["item_id"].startswith(mn.PHANTOM_ITEM_PREFIX) and r["name"] for r in new)
    assert len({r["item_id"] for r in rows}) == len(rows)   # spec: no duplicate item_id

    used = {i for _, i, _ in pairs_of(out / "raw/interactions.jsonl")}
    assert not used & {r["item_id"] for r in new}            # phantom items have no interactions


def test_auxs_phantom_appended(noisy_all):
    out, summary = noisy_all
    assert (out / "raw/auxs.csv").read_bytes().startswith((CLEAN_RAW / "auxs.csv").read_bytes())

    with open(out / "raw/auxs.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    new = rows[23_867:]
    assert summary["auxs.csv"] == {"clean_rows": 23_867, "noisy_rows": 23_897, "phantom_aux": 30}
    assert all(r["aux_id"].startswith(mn.PHANTOM_AUX_PREFIX) and r["name"] for r in new)
    assert {r["aux_type"] for r in new} == {"resource"}      # same rule as convert_book (res:)
    assert len({r["aux_id"] for r in rows}) == len(rows)     # spec: no duplicate aux_id


def test_untouched_files_copied_unchanged(noisy):
    out, _ = noisy
    assert sorted(p.name for p in (out / "raw").iterdir()) == sorted(mn.SPEC_FILES)
    injected = {name for name, _, _ in mn.INJECTORS}
    for name in set(mn.SPEC_FILES) - injected:
        assert (out / "raw" / name).read_bytes() == (CLEAN_RAW / name).read_bytes()


def test_deterministic(noisy, tmp_path):
    out, _ = noisy
    mn.make_noisy(CLEAN_RAW, tmp_path, seed=2020)
    for rel in ("raw/interactions.jsonl", "noisy_manifest.json"):
        assert (tmp_path / rel).read_bytes() == (out / rel).read_bytes()



@pytest.fixture(scope="module")
def book_data():
    return mn.load_raw(CLEAN_RAW)


def test_accepts_book_input(book_data):
    mn.check_book_assumptions(book_data)


def drop_orphan_items(rows, items):
    used = {r["item_id"] for r in rows}
    return [r for r in items if r["item_id"] in used]


@pytest.mark.parametrize("change, message", [
    (lambda rows, items: ([dict(rows[0], timestamp=1167609600)] + rows[1:], items), "columns"),
    (lambda rows, items: (rows[:100], items), "interactions"),
    (lambda rows, items: (rows, drop_orphan_items(rows, items)), "no item without interactions"),
])
def test_rejects_non_book_input(book_data, change, message):
    rows, items = change(book_data["interactions"], book_data["items"])
    with pytest.raises(SystemExit) as e:
        mn.check_book_assumptions({"interactions": rows, "items": items})
    assert message in str(e.value)
