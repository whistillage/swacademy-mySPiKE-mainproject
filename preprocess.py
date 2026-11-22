"""Preprocess a domain dataset: raw/ -> prep/<version>/ (preprocessing design v1.5).

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
    [7] IDs, [8] write    (not yet)

Standard library only (Python 3.10+).
"""

import argparse
import math
import random
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


# ---------------------------------------------------------------- driver

def preprocess(config: Config) -> State:
    config.check()
    data = run_validation(config.raw_dir)
    state = merge_duplicates(data)
    run_filters(state, config.user_min_interactions, config.item_min_interactions)
    split_users(state, config.seed, config.split)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
