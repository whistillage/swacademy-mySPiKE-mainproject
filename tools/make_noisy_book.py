"""Make a deliberately noisy copy of the book raw data, to test validation/preprocessing.

    python tools/make_noisy_book.py [clean_raw_dir] [out_domain_dir] [--seed 2020] [--force]

Defaults: data/book/raw -> data/book_noisy (run tools/convert_book.py first).

Writes:
    <out_domain_dir>/raw/                  the 5 spec files, with noise injected
    <out_domain_dir>/noisy_manifest.json   what was injected, per file (outside raw/)

Book only. The counts below are fixed for the book data, and the script relies on what
the converted book data looks like; check_book_assumptions() stops with a clear message
if the input does not match (other domains need their own noise script).

Each spec file has its own injector function (inject_<file>). Injectors run in a fixed
order on shared in-memory data, so a later injector can rely on an earlier one. Files that
no injector touched are copied byte for byte.

Implemented injectors
---------------------
interactions.jsonl (appended after the original rows):
  1. duplicate pairs   existing (user, item) pairs repeated with a different review
                       (most twice in total, some three times).
  2. small users       new users with 1-4 distinct items  -> removed by the 5-interaction filter.
  3. duplicate trap    new users with 4 distinct items + 1 repeated row = 5 rows
                       -> must still be removed (merge duplicates BEFORE filtering).
  New users' first item is an item with no interactions in the clean data, so those items
  must also disappear again (item filter runs AFTER the user filter).
  Expected: after preprocessing, the structural outputs (splits, ID lists, KG, counts) equal
  those of the clean data; only count/reviews of the duplicated pairs differ.
  Reviews are never empty: the spec requires a non-empty review (validation would stop).

relations.csv (appended):
  A new relation used only by the phantom items (see items.csv / kg.csv).
  Once those items are removed it has no triples left -> removed (KG cleanup step 4).
  Until the kg.csv injector exists it is simply an unused relation (a warning).

items.csv (appended):
  Phantom items with no interactions -> removed by the item filter. They are added after
  the interactions injector ran, so no fake user touches them.
  Intermediate state: they get their triples only from the kg.csv injector; until then
  they violate required condition 1 (every item has a 1-hop neighbour) -> validation error.

auxs.csv, kg.csv: not yet (copied unchanged).

Phantom items (planned across relations -> items -> auxs -> kg): items with no
interactions, plus aux / relations / triples that only they use, so that removing them
runs every KG cleanup step. Injectors share their names through data["phantom"].

Standard library only. Deterministic for a given seed.
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

SPEC_FILES = ("interactions.jsonl", "items.csv", "auxs.csv", "kg.csv", "relations.csv")

# interactions.jsonl
N_DUP_PAIRS = 500          # pairs that get extra rows
N_TRIPLE_PAIRS = 50        # ... of which appear three times in total
N_SMALL_USERS = 300        # 1-4 distinct items
N_TRAP_USERS = 100         # 4 distinct items + 1 duplicate row
NEW_USER_PREFIX = "NOISY_U"

# items.csv
N_PHANTOM_ITEMS = 50
PHANTOM_ITEM_PREFIX = "NOISY_I"

# relations.csv
PHANTOM_RELATIONS = [
    {"relation": "noisy:phantomLink", "head_type": "item", "tail_type": "aux",
     "sentence": "{head} is linked to {tails}", "reverse_sentence": "{tail} is linked to {head}"},
]


# ---------------------------------------------------------------- load / write

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_book_assumptions(data: dict):
    """Stop early if the input is not the converted book data this script was written for."""
    rows, items = data["interactions"], data["items"]
    keys = {k for r in rows for k in r}
    if keys != {"user_id", "item_id", "review"}:
        sys.exit(f"interactions.jsonl columns {sorted(keys)} != user_id/item_id/review "
                 "(book has no timestamp/rating). Is this the book data?")
    if len(rows) < 10 * N_DUP_PAIRS:
        sys.exit(f"only {len(rows)} interactions; book has 169,273. Is this the book data?")
    used = {r["item_id"] for r in rows}
    if all(r["item_id"] in used for r in items):
        sys.exit("no item without interactions; book has 221. Is this the book data?")
    if any(r["user_id"].startswith(NEW_USER_PREFIX) for r in rows):
        sys.exit(f"user IDs already use the prefix {NEW_USER_PREFIX!r}")
    if any(r["item_id"].startswith(PHANTOM_ITEM_PREFIX) for r in items):
        sys.exit(f"item IDs already use the prefix {PHANTOM_ITEM_PREFIX!r}")
    existing_relations = {r["relation"] for r in data["relations"]}
    if any(r["relation"] in existing_relations for r in PHANTOM_RELATIONS):
        sys.exit("phantom relation name already exists in relations.csv")


def load_raw(raw: Path) -> dict:
    """In-memory copy of the spec files that injectors read or modify."""
    return {
        "interactions": read_jsonl(raw / "interactions.jsonl"),
        "items": read_csv(raw / "items.csv"),
        "relations": read_csv(raw / "relations.csv"),
        "phantom": {},
    }


def write_csv(path: Path, rows):
    """Same CSV dialect as tools/convert_book.py, so unchanged rows stay byte-identical."""
    with open(path, encoding="utf-8", newline="", mode="w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows):
    with open(path, encoding="utf-8", newline="\n", mode="w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- injectors

def inject_interactions(data: dict, rng: random.Random) -> dict:
    """Duplicate pairs, users below 5 interactions, duplicate-trap users. Returns manifest part."""
    rows = data["interactions"]
    all_items = [r["item_id"] for r in data["items"]]

    used_items = {r["item_id"] for r in rows}
    orphan_items = sorted(i for i in all_items if i not in used_items)
    active_items = sorted(used_items)

    extra = []
    review_no = 0

    def review(kind):
        nonlocal review_no
        review_no += 1
        return f"(noisy {kind} #{review_no}) Synthetic review injected for testing."

    # 1. duplicate pairs
    dup_rows = rng.sample(range(len(rows)), N_DUP_PAIRS)
    triple = set(dup_rows[:N_TRIPLE_PAIRS])
    dup_pairs = []
    for idx in dup_rows:
        r = rows[idx]
        times = 2 if idx in triple else 1
        for _ in range(times):
            extra.append({"user_id": r["user_id"], "item_id": r["item_id"], "review": review("duplicate")})
        dup_pairs.append({"user_id": r["user_id"], "item_id": r["item_id"], "total_rows": times + 1})

    # 2./3. new users; first item is an orphan item (cycled), the rest are active items
    n_new = N_SMALL_USERS + N_TRAP_USERS
    orphan_cycle = iter(orphan_items * (1 + n_new // len(orphan_items)))
    new_users = []
    touched_orphans = set()

    def new_user_items(n_distinct):
        first = next(orphan_cycle)
        touched_orphans.add(first)
        rest = rng.sample([i for i in active_items if i != first], n_distinct - 1)
        return [first] + rest

    for k in range(N_SMALL_USERS):
        uid = f"{NEW_USER_PREFIX}{k:04d}"
        n = rng.randint(1, 4)
        for it in new_user_items(n):
            extra.append({"user_id": uid, "item_id": it, "review": review("small user")})
        new_users.append({"user_id": uid, "kind": "small", "distinct_items": n, "rows": n})

    for k in range(N_TRAP_USERS):
        uid = f"{NEW_USER_PREFIX}{N_SMALL_USERS + k:04d}"
        items = new_user_items(4)
        for it in items + [rng.choice(items)]:
            extra.append({"user_id": uid, "item_id": it, "review": review("trap user")})
        new_users.append({"user_id": uid, "kind": "duplicate_trap", "distinct_items": 4, "rows": 5})

    data["interactions"] = rows + extra

    summary = {
        "clean_rows": len(rows),
        "noisy_rows": len(rows) + len(extra),
        "extra_rows": len(extra),
        "duplicate_pairs": len(dup_pairs),
        "duplicate_extra_rows": sum(p["total_rows"] - 1 for p in dup_pairs),
        "new_users_small": N_SMALL_USERS,
        "new_users_duplicate_trap": N_TRAP_USERS,
        "new_user_rows": sum(u["rows"] for u in new_users),
        "orphan_items_clean": len(orphan_items),
        "orphan_items_touched": len(touched_orphans),
        "orphan_items_noisy": len(orphan_items) - len(touched_orphans),
    }
    return {
        "summary": summary,
        "duplicate_pairs": dup_pairs,
        "new_users": new_users,
        "touched_orphan_items": sorted(touched_orphans),
    }


def inject_relations(data: dict, rng: random.Random) -> dict:
    """Relation(s) used only by phantom items. Returns manifest part."""
    clean = len(data["relations"])
    new = [dict(r) for r in PHANTOM_RELATIONS]
    data["relations"] = data["relations"] + new
    data["phantom"]["relations"] = [r["relation"] for r in new]
    return {
        "summary": {"clean_rows": clean, "noisy_rows": clean + len(new), "phantom_relations": len(new)},
        "phantom_relations": new,
    }


def inject_items(data: dict, rng: random.Random) -> dict:
    """Phantom items without interactions. Returns manifest part."""
    clean = len(data["items"])
    new = [{"item_id": f"{PHANTOM_ITEM_PREFIX}{k:04d}", "name": f"Noisy Phantom Book {k:04d}"}
           for k in range(N_PHANTOM_ITEMS)]
    data["items"] = data["items"] + new
    data["phantom"]["items"] = [r["item_id"] for r in new]
    return {
        "summary": {"clean_rows": clean, "noisy_rows": clean + len(new), "phantom_items": len(new)},
        "phantom_items": [r["item_id"] for r in new],
    }


# (file name, injector, writer). Order matters: later injectors see earlier changes.
# interactions first: its fake users pick from the clean items only, never phantom items.
INJECTORS = (
    ("interactions.jsonl", inject_interactions,
     lambda data, path: write_jsonl(path, data["interactions"])),
    ("relations.csv", inject_relations,
     lambda data, path: write_csv(path, data["relations"])),
    ("items.csv", inject_items,
     lambda data, path: write_csv(path, data["items"])),
)


# ---------------------------------------------------------------- driver

def make_noisy(clean_raw: Path, out_dir: Path, seed: int = 2020) -> dict:
    """Returns {file name: summary} for the injected files."""
    rng = random.Random(seed)
    data = load_raw(clean_raw)
    check_book_assumptions(data)

    manifest = {"source": str(clean_raw), "seed": seed, "files": {}}
    for name, inject, _ in INJECTORS:
        manifest["files"][name] = inject(data, rng)

    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    writers = {name: write for name, _, write in INJECTORS}
    for name in SPEC_FILES:
        if name in writers:
            writers[name](data, raw / name)
        else:
            shutil.copyfile(clean_raw / name, raw / name)

    with open(out_dir / "noisy_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {name: part["summary"] for name, part in manifest["files"].items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("clean_raw", type=Path, nargs="?", default=Path("data/book/raw"),
                   help="converted book raw folder (default: data/book/raw)")
    p.add_argument("out", type=Path, nargs="?", default=Path("data/book_noisy"),
                   help="output domain folder (default: data/book_noisy)")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--force", action="store_true", help="overwrite existing output")
    args = p.parse_args(argv)

    if (args.out / "raw").exists() and not args.force:
        sys.exit(f"{args.out / 'raw'} already exists (use --force)")

    for name, summary in make_noisy(args.clean_raw, args.out, args.seed).items():
        print(f"[{name}]")
        for k, v in summary.items():
            print(f"  {k:>26}: {v:,}")


if __name__ == "__main__":
    main()
