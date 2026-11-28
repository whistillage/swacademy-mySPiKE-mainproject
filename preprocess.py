"""Preprocess a domain dataset: raw/ -> prep/<version>/ (preprocessing design v1.10).

    python preprocess.py data/<domain> --version v1 [--seed 2020] [--split 0.7 0.1 0.2]
                         [--user-min 5] [--item-min 1]

Flow (design chapter 3):
    [1] validate          validate_dataset.validate(); stop on ERROR
    [2] merge duplicates  one entry per (user, item); KG exact duplicates collapse
    [3] user filter       drop users with < user_min distinct items
    [4] item filter       drop items with < item_min users, then clean the KG (4 steps)
    [5] isolated items    drop items that appear in no KG triple (head or tail)
        repeat [3]-[5] until a round removes no interaction in [4] or [5]
    [6] split             per user: test = max(1, R(0.2 n)), valid = max(1, R(0.1 n)), rest train
    [7] assign IDs        0-based, by sorted original ID; items then aux share node_idx
    [8] write             prep/<version>/: 10 files (design chapter 8), via a temporary folder

Standard library only (Python 3.10+).
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import validate_dataset as vd


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    """Design chapter 9. Everything here is recorded in meta.json."""
    domain_dir: Path
    version: str
    seed: int = 2020
    split: tuple = (0.7, 0.1, 0.2)        # train / valid / test
    rounding: str = "half_up"             # fixed
    user_min_interactions: int = 5
    item_min_interactions: int = 1

    @property
    def raw_dir(self) -> Path:
        return self.domain_dir / "raw"

    @property
    def out_dir(self) -> Path:
        return self.domain_dir / "prep" / self.version

    def check(self):
        if len(self.split) != 3 or any(not 0 < x < 1 for x in self.split) or abs(sum(self.split) - 1) > 1e-9:
            raise ValueError(f"split must be three ratios in (0, 1) summing to 1: {self.split}")
        if self.user_min_interactions < 3:
            raise ValueError("user_min_interactions must be >= 3 (train / valid / test need one each)")
        if self.item_min_interactions < 1:
            raise ValueError("item_min_interactions must be >= 1")


# meta.json "stats": always all keys, in this order, 0 when nothing happened (design 8.9)
STATS_KEYS = (
    "duplicate_pairs_merged",      # pairs that had more than one row
    "duplicate_rows_merged",       # rows that disappeared by merging
    "kg_duplicate_rows_removed",
    "users_removed",
    "pairs_removed",               # interactions removed by the filters
    "items_removed",               # item filter (too few users)
    "isolated_items_removed",      # in no KG triple
    "aux_removed",
    "kg_triples_removed",
    "relations_removed",
    "filter_rounds",
    "split_pairs",                 # {"train": n, "valid": n, "test": n}
)

OUTPUT_FILES = ("train.txt", "valid.txt", "test.txt", "user_list.csv", "item_list.csv", "aux_list.csv",
                "relation_list.csv", "kg.csv", "interactions.jsonl", "meta.json")


# ---------------------------------------------------------------- working state

@dataclass
class State:
    """Data being preprocessed. IDs are the stripped original strings (design 5, ID cleanup)."""
    pairs: dict            # (user_id, item_id) -> {"count": int, "reviews": [{"text", ["timestamp"], ["rating"]}]}
    items: dict            # item_id -> name
    auxs: dict             # aux_id -> (name, aux_type)
    relations: dict        # relation -> row dict (head_type, tail_type, sentence, reverse_sentence)
    kg: set                # {(head_type, head, relation, tail_type, tail)}, exact duplicates removed
    optional_keys: tuple   # timestamp / rating present in the input
    stats: dict = field(default_factory=dict)
    split: dict = None     # user_id -> {"train": [item_id], "valid": [...], "test": [...]}
    ids: "Ids" = None


@dataclass
class Ids:
    """Design 7: *_idx are 0-based numbers in sorted order of the original ID strings
    (Unicode code points). Items and aux share one node space, items first."""
    user: dict       # user_id -> user_idx
    item: dict       # item_id -> item_idx (= node_idx)
    aux: dict        # aux_id -> aux_idx
    relation: dict   # relation -> rel_idx

    @property
    def n_items(self) -> int:
        return len(self.item)

    def node(self, node_type: str, node_id: str) -> int:
        """node_idx of an item or aux."""
        return self.item[node_id] if node_type == "item" else self.n_items + self.aux[node_id]


# ---------------------------------------------------------------- [1] validate

class ValidationFailed(Exception):
    def __init__(self, report):
        super().__init__("validation failed")
        self.report = report


def run_validation(raw_dir: Path):
    """Design 4.1: stop with the full report on ERROR, otherwise only count the warnings."""
    report, data = vd.validate(raw_dir)
    if report.has_errors:
        raise ValidationFailed(report)
    warnings = report.by_level(vd.WARNING)
    print(f"[1] 검증 통과: 경고 {len(warnings)}종 {sum(w.count for w in warnings):,}건 "
          f"(자세한 내용: python validate_dataset.py {raw_dir})")
    return data


# ---------------------------------------------------------------- [2] merge duplicates

def merge_duplicates(data: vd.Dataset) -> State:
    """Design 5.1 / 5.2. Reviews keep input order, one element per merged row."""
    pairs = {}
    for r in data.interactions:
        review = {"text": r["review"]}
        for key in data.optional_keys:
            review[key] = r.get(key)
        entry = pairs.setdefault((r["user_id"], r["item_id"]), {"count": 0, "reviews": []})
        entry["count"] += 1
        entry["reviews"].append(review)

    state = State(
        pairs=pairs,
        items={r["item_id"]: r["name"] for r in data.items},
        auxs={r["aux_id"]: (r["name"], r["aux_type"].strip()) for r in data.auxs},
        relations={r["relation"]: {k: (r[k].strip() if k.endswith("_type") else r[k])
                                   for k in ("head_type", "tail_type", "sentence", "reverse_sentence")}
                   for r in data.relations},
        kg=set(data.kg_triples),           # validation passed: every format-valid triple is here
        optional_keys=data.optional_keys,
    )
    state.stats["duplicate_pairs_merged"] = sum(e["count"] > 1 for e in pairs.values())
    state.stats["duplicate_rows_merged"] = len(data.interactions) - len(pairs)
    state.stats["kg_duplicate_rows_removed"] = len(data.kg_rows) - len(state.kg)
    print(f"[2] 중복 합치기: 상호작용 {len(data.interactions):,}행 → {len(pairs):,}쌍 "
          f"(중복 쌍 {state.stats['duplicate_pairs_merged']:,}), "
          f"KG {len(data.kg_rows):,}행 → {len(state.kg):,}트리플")
    return state


# ---------------------------------------------------------------- [3] / [4] filters

def add_stat(state: State, key: str, n: int):
    state.stats[key] = state.stats.get(key, 0) + n


def filter_users(state: State, user_min: int) -> int:
    """Design 5.3: drop users with fewer than user_min distinct items (pairs are already merged)."""
    per_user = {}
    for u, _ in state.pairs:
        per_user[u] = per_user.get(u, 0) + 1
    removed = {u for u, n in per_user.items() if n < user_min}
    before = len(state.pairs)
    state.pairs = {k: v for k, v in state.pairs.items() if k[0] not in removed}
    add_stat(state, "users_removed", len(removed))
    add_stat(state, "pairs_removed", before - len(state.pairs))
    return len(removed)


def remove_items(state: State, removed: set, stat: str = "items_removed"):
    """Remove items (and their pairs), then clean the KG (design 5.4):
    1. triples that contain a removed item
    2. aux no longer linked to a remaining item (tail of item -> aux)
    3. triples that contain an aux removed in 2 (aux -> aux)
    4. relations without triples
    """
    if not removed:
        return
    before_pairs, before_kg = len(state.pairs), len(state.kg)
    for i in removed:
        del state.items[i]
    state.pairs = {k: v for k, v in state.pairs.items() if k[1] not in removed}

    def has_removed_item(t):
        return (t[0] == "item" and t[1] in removed) or (t[3] == "item" and t[4] in removed)
    state.kg = {t for t in state.kg if not has_removed_item(t)}                       # 1
    linked = {t[4] for t in state.kg if t[0] == "item" and t[3] == "aux"}
    dead_aux = set(state.auxs) - linked                                                # 2
    for a in dead_aux:
        del state.auxs[a]
    state.kg = {t for t in state.kg
                if not ((t[0] == "aux" and t[1] in dead_aux) or (t[3] == "aux" and t[4] in dead_aux))}  # 3
    used = {t[2] for t in state.kg}
    dead_rel = set(state.relations) - used                                             # 4
    for r in dead_rel:
        del state.relations[r]

    add_stat(state, stat, len(removed))
    add_stat(state, "pairs_removed", before_pairs - len(state.pairs))
    add_stat(state, "aux_removed", len(dead_aux))
    add_stat(state, "kg_triples_removed", before_kg - len(state.kg))
    add_stat(state, "relations_removed", len(dead_rel))


def filter_items(state: State, item_min: int) -> int:
    """Design 5.4: drop items with fewer than item_min users (default 1: no interactions)."""
    per_item = dict.fromkeys(state.items, 0)
    for _, i in state.pairs:
        per_item[i] += 1
    removed = {i for i, n in per_item.items() if n < item_min}
    remove_items(state, removed)
    return len(removed)


def remove_isolated_items(state: State) -> int:
    """Design 5.5: drop items that appear in no triple. Being only the tail of item -> item is
    enough to stay (the training code adds reverse triples)."""
    in_kg = {t[1] for t in state.kg if t[0] == "item"} | {t[4] for t in state.kg if t[3] == "item"}
    removed = set(state.items) - in_kg
    remove_items(state, removed, stat="isolated_items_removed")
    return len(removed)


def run_filters(state: State, user_min: int, item_min: int) -> int:
    """Design 5.6: repeat [3] -> [4] -> [5]. Only an interaction removed in [4] or [5] can give the
    user filter new work, so stop after a round in which [4] and [5] removed none.
    Returns the number of rounds."""
    rounds = 0
    while True:
        rounds += 1
        users = filter_users(state, user_min)
        pairs_before = len(state.pairs)
        items = filter_items(state, item_min)
        isolated = remove_isolated_items(state)
        print(f"[3-5] 회차 {rounds}: 사용자 {users:,}명, 아이템 {items:,}개, 고립 아이템 {isolated:,}개 제거 "
              f"→ 사용자 {len({u for u, _ in state.pairs}):,}명, 아이템 {len(state.items):,}, "
              f"aux {len(state.auxs):,}, 관계 {len(state.relations):,}, KG {len(state.kg):,}, 쌍 {len(state.pairs):,}")
        if len(state.pairs) == pairs_before:
            break
    state.stats["filter_rounds"] = rounds
    return rounds


# ---------------------------------------------------------------- [6] split

def half_up(x: Fraction) -> int:
    """Round half up (0.5 -> 1). Python's round() rounds half to even."""
    return math.floor(x + Fraction(1, 2))


def split_counts(n: int, split) -> tuple:
    """Design 6.1: (train, valid, test) for a user with n items. Ratios are used as exact
    fractions (0.1 * 15 is 1.5000000000000002 in floating point)."""
    _, p_valid, p_test = (Fraction(str(x)) for x in split)
    test = max(1, half_up(p_test * n))
    valid = max(1, half_up(p_valid * n))
    train = n - test - valid
    if train < 1:
        raise ValueError(f"split {tuple(split)} leaves no train item for a user with {n} items")
    return train, valid, test


def split_user(user_id: str, items, seed: int, split) -> dict:
    """Sort the user's items by item_id, shuffle with the user's own generator seeded by
    "<seed>:<user_id>", then take test, valid, and the rest as train (design 6.1)."""
    order = sorted(items)
    random.Random(f"{seed}:{user_id}").shuffle(order)
    _, valid, test = split_counts(len(order), split)
    return {"test": order[:test], "valid": order[test:test + valid], "train": order[test + valid:]}


def split_users(state: State, seed: int, split) -> dict:
    per_user = {}
    for u, i in state.pairs:
        per_user.setdefault(u, []).append(i)
    state.split = {u: split_user(u, items, seed, split) for u, items in per_user.items()}
    counts = {part: sum(len(s[part]) for s in state.split.values()) for part in ("train", "valid", "test")}
    state.stats["split_pairs"] = counts
    total = sum(counts.values()) or 1
    print("[6] 분할: " + ", ".join(f"{k} {v:,} ({v / total:.1%})" for k, v in counts.items()))
    return counts


# ---------------------------------------------------------------- [7] IDs

def numbering(ids) -> dict:
    return {x: k for k, x in enumerate(sorted(ids))}


def assign_ids(state: State) -> Ids:
    state.ids = Ids(
        user=numbering({u for u, _ in state.pairs}),
        item=numbering(state.items),
        aux=numbering(state.auxs),
        relation=numbering(state.relations),
    )
    n_items, n_aux = len(state.ids.item), len(state.ids.aux)
    print(f"[7] ID 부여: 사용자 {len(state.ids.user):,}, 아이템 {n_items:,} (node 0~{n_items - 1}), "
          f"aux {n_aux:,} (node {n_items}~{n_items + n_aux - 1}), 관계 {len(state.ids.relation):,}")
    return state.ids


# ---------------------------------------------------------------- [8] write

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def build_meta(config: Config, state: State) -> dict:
    ids = state.ids
    n_items, n_aux = ids.n_items, len(ids.aux)
    return {
        "domain": config.domain_dir.name,
        "version": config.version,
        "n_users": len(ids.user),
        "n_items": n_items,
        "n_aux": n_aux,
        "n_relations": len(ids.relation),
        "node_ranges": {"item": [0, n_items - 1], "aux": [n_items, n_items + n_aux - 1]},
        "config": {
            "seed": config.seed,
            "split": list(config.split),
            "rounding": config.rounding,
            "user_min_interactions": config.user_min_interactions,
            "item_min_interactions": config.item_min_interactions,
        },
        "stats": {k: state.stats.get(k, 0) for k in STATS_KEYS},
        "input_hash": {name: file_hash(config.raw_dir / name) for name in vd.FILES},
    }


def write_csv(path: Path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_outputs(config: Config, state: State, out: Path):
    """Write the 10 files of design chapter 8 into `out`. Everything is sorted and uses "\n"."""
    ids = state.ids
    by_idx = lambda mapping: sorted(mapping, key=mapping.get)

    for part in ("train", "valid", "test"):
        with open(out / f"{part}.txt", "w", encoding="utf-8", newline="\n") as f:
            for u in by_idx(ids.user):
                items = sorted(ids.item[i] for i in state.split[u][part])
                f.write(" ".join(map(str, [ids.user[u]] + items)) + "\n")

    write_csv(out / "user_list.csv", ("user_id", "user_idx"), ((u, ids.user[u]) for u in by_idx(ids.user)))
    write_csv(out / "item_list.csv", ("item_id", "item_idx", "node_idx", "name"),
              ((i, ids.item[i], ids.node("item", i), state.items[i]) for i in by_idx(ids.item)))
    write_csv(out / "aux_list.csv", ("aux_id", "aux_idx", "node_idx", "name", "aux_type"),
              ((a, ids.aux[a], ids.node("aux", a), *state.auxs[a]) for a in by_idx(ids.aux)))
    write_csv(out / "relation_list.csv",
              ("relation", "rel_idx", "head_type", "tail_type", "sentence", "reverse_sentence"),
              ((r, ids.relation[r], *(state.relations[r][k] for k in ("head_type", "tail_type", "sentence",
                                                                     "reverse_sentence")))
               for r in by_idx(ids.relation)))
    write_csv(out / "kg.csv", ("head", "relation", "tail"),
              sorted((ids.node(ht, h), ids.relation[rel], ids.node(tt, t)) for ht, h, rel, tt, t in state.kg))

    with open(out / "interactions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (u, i) in sorted(state.pairs, key=lambda k: (ids.user[k[0]], ids.item[k[1]])):
            e = state.pairs[(u, i)]
            row = {"user_idx": ids.user[u], "item_idx": ids.item[i], "count": e["count"], "reviews": e["reviews"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(build_meta(config, state), ensure_ascii=False, indent=2) + "\n")


def check_output_dir(config: Config):
    """Design 2: never overwrite an existing version folder."""
    if config.out_dir.exists():
        raise FileExistsError(f"{config.out_dir} already exists (use a new --version or delete the folder)")


def write_prep(config: Config, state: State):
    """Write into prep/.<version>.tmp, then rename it to prep/<version>, so a failed run never
    leaves a half-written version folder."""
    check_output_dir(config)
    tmp = config.out_dir.parent / f".{config.version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)          # leftover of an interrupted run of this script
    tmp.mkdir(parents=True)
    write_outputs(config, state, tmp)
    tmp.rename(config.out_dir)
    print(f"[8] 출력: {config.out_dir} (파일 {len(OUTPUT_FILES)}개)")


# ---------------------------------------------------------------- driver

def preprocess(config: Config, write: bool = True) -> State:
    """Run [1]-[8]. write=False stops after [7] (nothing is written)."""
    config.check()
    if write:
        check_output_dir(config)    # fail before the work, not after
    data = run_validation(config.raw_dir)
    state = merge_duplicates(data)
    run_filters(state, config.user_min_interactions, config.item_min_interactions)
    split_users(state, config.seed, config.split)
    assign_ids(state)
    if write:
        write_prep(config, state)
    return state


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="SPiKE preprocessing: data/<domain>/raw -> data/<domain>/prep/<version>")
    p.add_argument("domain_dir", type=Path, help="domain folder, e.g. data/book")
    p.add_argument("--version", required=True, help="output version, e.g. v1")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--split", type=float, nargs=3, default=(0.7, 0.1, 0.2), metavar=("TRAIN", "VALID", "TEST"))
    p.add_argument("--user-min", type=int, default=5, help="minimum interactions per user (default 5)")
    p.add_argument("--item-min", type=int, default=1, help="minimum interactions per item (default 1)")
    a = p.parse_args(argv)
    return Config(domain_dir=a.domain_dir, version=a.version, seed=a.seed, split=tuple(a.split),
                  user_min_interactions=a.user_min, item_min_interactions=a.item_min)


def main(argv=None) -> int:
    config = parse_args(argv)
    try:
        preprocess(config)
    except ValidationFailed as e:
        print(e.report.render())
        print("\n검증 에러가 있어 전처리를 중단합니다.")
        return 1
    except ValueError as e:
        print(f"설정 오류: {e}")
        return 2
    except FileExistsError as e:
        print(f"출력 폴더 오류: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
