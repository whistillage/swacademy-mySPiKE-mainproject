"""Preprocess a domain dataset: raw/ -> prep/<version>/ (preprocessing design v1.5).

    python preprocess.py data/<domain> --version v1 [--seed 2020] [--split 0.7 0.1 0.2]
                         [--user-min 5] [--item-min 1]

Flow (design chapter 3):
    [1] validate          validate_dataset.validate(); stop on ERROR
    [2] merge duplicates  one entry per (user, item); KG exact duplicates collapse
    [3]-[5] filters       (not yet)
    [6] split, [7] IDs, [8] write   (not yet)

Standard library only (Python 3.10+).
"""

import argparse
import sys
from dataclasses import dataclass, field
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
        if self.user_min_interactions < 1 or self.item_min_interactions < 1:
            raise ValueError("minimum interaction counts must be >= 1")


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


# ---------------------------------------------------------------- driver

def preprocess(config: Config) -> State:
    config.check()
    data = run_validation(config.raw_dir)
    state = merge_duplicates(data)
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
