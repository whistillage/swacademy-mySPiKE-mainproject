"""Build profiles from a prep version: prep/<version>/ -> profile/<version>_<method>/ (LLM profile design v1.0).

    python make_profiles.py data/<domain> --prep v1 --config configs/profile/<method>.json --stage <stage>

    <stage> = item_raw | item_llm | aux_raw | aux_llm | user_raw | user_llm | embed | all

The command line takes only paths and the stage; every other setting lives in the config file
(design chapter 9). The output folder name is <prep version>_<config file name> (design 8.1).

Implemented stages: none yet (item_raw is being added).

Standard library only (Python 3.10+) for the raw-profile stages.
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------- stages

STAGES = ("item_raw", "item_llm", "aux_raw", "aux_llm", "user_raw", "user_llm", "embed", "all")

# stage -> function(run). Filled in as stages are implemented.
STAGE_RUNNERS = {}


# ---------------------------------------------------------------- config file

# Design chapter 9. Every key is required; unknown keys are an error (a typo must not fall back
# silently to some default). Keys are added here as the stages that use them are implemented.
CONFIG_KEYS = {
    "seed": int,               # random selection seed (design 4.3)
    "item_review_cap": int,    # max reviews in an item raw profile (design 5.1)
}

METHOD_NAME = re.compile(r"[a-z0-9-]+")   # design 8.1: '_' separates prep version and method


@dataclass(frozen=True)
class Settings:
    """Contents of configs/profile/<method>.json."""
    seed: int
    item_review_cap: int

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in CONFIG_KEYS}


def load_settings(path: Path) -> Settings:
    """Read and check a config file. Raises ValueError with a message for the user."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"설정 파일이 없습니다: {path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"설정 파일이 올바른 JSON이 아닙니다: {path} ({e})") from None
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일은 JSON 객체여야 합니다: {path}")

    unknown = sorted(set(data) - set(CONFIG_KEYS))
    missing = [k for k in CONFIG_KEYS if k not in data]
    if unknown:
        raise ValueError(f"설정 파일에 알 수 없는 키가 있습니다: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"설정 파일에 빠진 키가 있습니다: {', '.join(missing)}")
    for key, typ in CONFIG_KEYS.items():
        value = data[key]
        # bool is a subclass of int in Python; true/false is not a valid number here
        if not isinstance(value, typ) or isinstance(value, bool):
            raise ValueError(f"설정 키 {key}의 값은 {typ.__name__}이어야 합니다: {value!r}")

    if data["item_review_cap"] < 0:
        raise ValueError(f"item_review_cap은 0 이상이어야 합니다: {data['item_review_cap']}")
    return Settings(**data)


# ---------------------------------------------------------------- prep input

class PrepError(Exception):
    """prep/<version>/ is missing a file or is inconsistent (exit code 3)."""


@dataclass(frozen=True)
class Relation:
    name: str
    head_type: str          # "item" or "aux"
    tail_type: str
    sentence: str           # {head} ... {tails}
    reverse_sentence: str   # {tail} ... {head}


@dataclass
class Prep:
    """What the item raw profiles need from prep/<version>/. Lists are indexed by *_idx."""
    meta: dict
    item_ids: list          # item_idx -> item_id
    item_names: list        # item_idx -> name
    aux_names: list         # aux_idx -> name
    user_ids: list          # user_idx -> user_id
    relations: list         # rel_idx -> Relation
    kg: list                # [(head node_idx, rel_idx, tail node_idx)] as in kg.csv
    item_reviews: dict      # item_idx -> [(user_idx, [review text, ...])], train pairs only

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def is_item(self, node: int) -> bool:
        return node < self.n_items

    def node_name(self, node: int) -> str:
        return self.item_names[node] if self.is_item(node) else self.aux_names[node - self.n_items]


def read_csv_rows(path: Path) -> list:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        raise PrepError(f"prep 파일이 없습니다: {path}") from None


def check_indexed(rows: list, idx_key: str, expected: int, path: Path):
    """Rows must be in *_idx order 0..n-1 and their number must match meta.json."""
    if len(rows) != expected:
        raise PrepError(f"{path.name}: 행 수 {len(rows)}가 meta.json의 개수 {expected}와 다릅니다")
    for pos, row in enumerate(rows):
        if int(row[idx_key]) != pos:
            raise PrepError(f"{path.name}: {pos + 2}행의 {idx_key}가 {row[idx_key]}입니다 ({pos}이어야 함)")


def load_prep(prep_dir: Path) -> Prep:
    """Read prep/<version>/ with light consistency checks (counts, contiguous *_idx, node and
    relation ranges, every train pair present in interactions.jsonl). Keeps only the review text
    of train pairs: valid / test reviews must not reach the profiles (design 1.4, 5.1)."""
    try:
        with open(prep_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    except FileNotFoundError:
        raise PrepError(f"prep 파일이 없습니다: {prep_dir / 'meta.json'}") from None
    n_items, n_aux = meta["n_items"], meta["n_aux"]

    items = read_csv_rows(prep_dir / "item_list.csv")
    check_indexed(items, "item_idx", n_items, prep_dir / "item_list.csv")
    auxs = read_csv_rows(prep_dir / "aux_list.csv")
    check_indexed(auxs, "aux_idx", n_aux, prep_dir / "aux_list.csv")
    if any(int(r["node_idx"]) != int(r["item_idx"]) for r in items) or \
            any(int(r["node_idx"]) != n_items + int(r["aux_idx"]) for r in auxs):
        raise PrepError("item_list.csv / aux_list.csv: node_idx가 규칙(아이템 먼저, aux는 아이템 수 + aux_idx)과 다릅니다")
    users = read_csv_rows(prep_dir / "user_list.csv")
    check_indexed(users, "user_idx", meta["n_users"], prep_dir / "user_list.csv")
    rels = read_csv_rows(prep_dir / "relation_list.csv")
    check_indexed(rels, "rel_idx", meta["n_relations"], prep_dir / "relation_list.csv")
    relations = [Relation(r["relation"], r["head_type"], r["tail_type"], r["sentence"], r["reverse_sentence"])
                 for r in rels]

    n_nodes = n_items + n_aux
    kg = [(int(r["head"]), int(r["relation"]), int(r["tail"])) for r in read_csv_rows(prep_dir / "kg.csv")]
    for h, rel, t in kg:
        if not (0 <= h < n_nodes and 0 <= t < n_nodes and 0 <= rel < len(relations)):
            raise PrepError(f"kg.csv: 범위를 벗어난 값이 있습니다 ({h}, {rel}, {t})")

    train = set()
    try:
        with open(prep_dir / "train.txt", encoding="utf-8") as f:
            for line in f:
                u, *its = map(int, line.split())
                if not 0 <= u < len(users) or any(not 0 <= i < n_items for i in its):
                    raise PrepError(f"train.txt: 범위를 벗어난 번호가 있습니다 (user_idx {u})")
                train.update((u, i) for i in its)
    except FileNotFoundError:
        raise PrepError(f"prep 파일이 없습니다: {prep_dir / 'train.txt'}") from None

    item_reviews, found = {}, 0
    try:
        with open(prep_dir / "interactions.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                u, i = row["user_idx"], row["item_idx"]
                if (u, i) in train:
                    found += 1
                    item_reviews.setdefault(i, []).append((u, [r["text"] for r in row["reviews"]]))
    except FileNotFoundError:
        raise PrepError(f"prep 파일이 없습니다: {prep_dir / 'interactions.jsonl'}") from None
    if found != len(train):
        raise PrepError(f"interactions.jsonl: train 쌍 {len(train) - found}개가 없습니다")

    return Prep(meta=meta,
                item_ids=[r["item_id"] for r in items], item_names=[r["name"] for r in items],
                aux_names=[r["name"] for r in auxs], user_ids=[r["user_id"] for r in users],
                relations=relations, kg=kg, item_reviews=item_reviews)


# ---------------------------------------------------------------- run

@dataclass(frozen=True)
class Run:
    """One invocation: where to read, where to write, which stage, with which settings."""
    domain_dir: Path
    prep_version: str
    config_path: Path
    stage: str
    settings: Settings

    @property
    def method(self) -> str:
        return self.config_path.stem

    @property
    def prep_dir(self) -> Path:
        return self.domain_dir / "prep" / self.prep_version

    @property
    def out_dir(self) -> Path:
        return self.domain_dir / "profile" / f"{self.prep_version}_{self.method}"


def check_run(run: Run):
    """Checks that need no prep data. Raises ValueError."""
    if run.config_path.suffix != ".json":
        raise ValueError(f"설정 파일은 .json이어야 합니다: {run.config_path}")
    if not METHOD_NAME.fullmatch(run.method):
        raise ValueError(f"설정 파일 이름은 영어 소문자, 숫자, '-'만 쓸 수 있습니다: {run.config_path.name}")
    if not (run.prep_dir / "meta.json").is_file():
        raise ValueError(f"prep 폴더가 없거나 meta.json이 없습니다: {run.prep_dir}")
    if run.stage not in STAGE_RUNNERS:
        raise ValueError(f"아직 구현되지 않은 단계입니다: {run.stage}")


# ---------------------------------------------------------------- driver

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPiKE profiles: data/<domain>/prep/<ver> -> data/<domain>/profile/<ver>_<method>")
    p.add_argument("domain_dir", type=Path, help="domain folder, e.g. data/book")
    p.add_argument("--prep", required=True, help="prep version to read, e.g. v1")
    p.add_argument("--config", required=True, type=Path, help="configs/profile/<method>.json")
    p.add_argument("--stage", required=True, choices=STAGES)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Exit codes: 0 success, 2 config error, 3 prep / output / tracking error."""
    a = parse_args(argv)
    try:
        run = Run(domain_dir=a.domain_dir, prep_version=a.prep, config_path=a.config, stage=a.stage,
                  settings=load_settings(a.config))
        check_run(run)
    except ValueError as e:
        print(f"설정 오류: {e}")
        return 2
    try:
        STAGE_RUNNERS[run.stage](run)
    except PrepError as e:
        print(f"prep 오류: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
