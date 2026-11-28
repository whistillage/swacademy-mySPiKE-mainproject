"""Build profiles from a prep version: prep/<version>/ -> profile/<version>_<method>/ (LLM profile design v1.1).

    python make_profiles.py data/<domain> --prep v1 --config configs/profile/<method>.json --stage <stage>

    <stage> = item_raw | item_llm | aux_raw | aux_llm | user_raw | user_llm | embed | all

The command line takes only paths and the stage; every other setting lives in the config file
(design chapter 9). The output folder name is <prep version>_<config file name> (design 8.1).

Implemented stages: item_raw.

Output (design chapter 8):
    profile/<version>_<method>/raw/item_raw_profiles.jsonl   {"item_idx", "item_id", "text"} per line
    profile/<version>_<method>/meta.json                     prep hash, per-stage config / inputs / stats
    profile/<version>_<method>/profile.log                   progress lines

A finished stage (its output file exists) is skipped after checking it still matches the prep,
the config keys it used and its input files; the code never deletes anything (design chapter 10).

Standard library only (Python 3.10+) for the raw-profile stages.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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


# ---------------------------------------------------------------- sentences and sampling

def quote(name: str) -> str:
    """Design 4.1: names in single quotes. Quotes inside the name are kept as they are."""
    return f"'{name}'"


def join_names(quoted: list) -> str:
    """'A' / 'A', and 'B' / 'A', 'B', and 'C' (design 4.1: the author's format, comma even for two)."""
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]


def fill(template: str, **values: str) -> str:
    """Fill {head} / {tails} / {tail} and end with one '.'. A '.' already at the end of the template
    (validate_dataset only warns about it) is dropped first so it never doubles.
    str.replace, not str.format: other braces in a template must not break it."""
    text = template.rstrip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text + "."


def name_order(prep: Prep, node: int) -> tuple:
    """Sort key: name by Unicode code point (design 4.1), node_idx breaks ties between equal names."""
    return prep.node_name(node), node


def forward_sentences(prep: Prep, head: int, edges: list) -> list:
    """Sentences for triples with this head: one per relation, its tails joined (design 4.1).
    edges: [(rel_idx, tail node_idx)]. Relations in rel_idx order, tails in name order."""
    by_rel = {}
    for rel, tail in edges:
        by_rel.setdefault(rel, []).append(tail)
    head_name = quote(prep.node_name(head))
    return [fill(prep.relations[rel].sentence, head=head_name,
                 tails=join_names([quote(prep.node_name(t)) for t in sorted(tails, key=lambda n: name_order(prep, n))]))
            for rel, tails in sorted(by_rel.items())]


def reverse_sentences(prep: Prep, tail: int, edges: list) -> list:
    """Sentences for triples with this tail: one per triple (design 4.1).
    edges: [(head node_idx, rel_idx)]. Relations in rel_idx order, heads in name order."""
    tail_name = quote(prep.node_name(tail))
    return [fill(prep.relations[rel].reverse_sentence, tail=tail_name, head=quote(prep.node_name(head)))
            for head, rel in sorted(edges, key=lambda e: (e[1], name_order(prep, e[0])))]


def entity_rng(seed: int, kind: str, entity_id: str) -> random.Random:
    """Design 4.3: one generator per entity, seeded by the string "<seed>:<kind>:<original ID>".
    A str seed does not depend on PYTHONHASHSEED, the processing order or other entities."""
    return random.Random(f"{seed}:{kind}:{entity_id}")


def pick(candidates: list, cap: int, rng: random.Random) -> list:
    """Design 4.3: candidates come already sorted. Up to `cap` of them: all, in that order, when
    there are no more than `cap`; otherwise rng.sample, in the order sample returns."""
    if len(candidates) <= cap:
        return list(candidates)
    return rng.sample(candidates, cap)


# ---------------------------------------------------------------- item raw profiles

def review_units(prep: Prep, item: int) -> list:
    """Design 5.1: every review on a train pair of this item is one candidate, sorted by
    (the reviewer's user_id, position in that pair's reviews list). Only the text is used."""
    pairs = sorted(prep.item_reviews.get(item, []), key=lambda e: prep.user_ids[e[0]])
    return [text for _, texts in pairs for text in texts]


def item_raw_profile(prep: Prep, item: int, forward: list, reverse: list, seed: int, review_cap: int) -> str:
    """Design 5.1: forward sentences, reverse sentences, reviews, joined by one space.
    forward: [(rel_idx, tail)] of triples with this item as head (item -> aux, item -> item).
    reverse: [(head item, rel_idx)] of item -> item triples with this item as tail.
    Reviews are used as they are: no cutting, quotes kept (the 6.4 trimming belongs to the LLM stage)."""
    parts = forward_sentences(prep, item, forward) + reverse_sentences(prep, item, reverse)
    rng = entity_rng(seed, "item", prep.item_ids[item])
    reviews = pick(review_units(prep, item), review_cap, rng)
    if reviews:
        parts.append("Some users say: " + ", ".join(quote(r) for r in reviews))
    if not parts:
        # every item is in at least one triple (spec required condition 1, preprocessing 5.5)
        raise PrepError(f"item {prep.item_ids[item]}: 트리플도 train 리뷰도 없어 프로필을 만들 수 없습니다")
    return " ".join(parts)


def build_item_raw_profiles(prep: Prep, seed: int, review_cap: int) -> list:
    """Item raw profile texts, indexed by item_idx."""
    forward = [[] for _ in range(prep.n_items)]
    reverse = [[] for _ in range(prep.n_items)]
    for head, rel, tail in prep.kg:
        if prep.is_item(head):
            forward[head].append((rel, tail))
            if prep.is_item(tail):
                reverse[tail].append((head, rel))
    return [item_raw_profile(prep, i, forward[i], reverse[i], seed, review_cap) for i in range(prep.n_items)]


# ---------------------------------------------------------------- output and tracking

class TrackingError(Exception):
    """The profile folder does not match the current prep / config / files (exit code 3)."""


# Config keys each stage depends on. Only these are recorded and compared for the stage, so adding
# keys for later stages to the config file does not invalidate finished stages.
STAGE_CONFIG_KEYS = {
    "item_raw": ("seed", "item_review_cap"),
}

ITEM_RAW_INPUTS = ("item_list.csv", "aux_list.csv", "user_list.csv", "relation_list.csv", "kg.csv",
                   "train.txt", "interactions.jsonl")
ITEM_RAW_FILE = Path("raw") / "item_raw_profiles.jsonl"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    """Commit of this code and whether it has uncommitted changes; None when git is not available."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                               check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def percentiles(values: list) -> dict:
    """Nearest-rank median / 90% / 99% / max of a non-empty list."""
    v = sorted(values)
    at = lambda q: v[max(0, math.ceil(q * len(v)) - 1)]
    return {"median": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": v[-1]}


def write_atomic(path: Path, text: str):
    """Write to a temporary file next to `path`, then rename: `path` is never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)


class Log:
    """Progress lines on the screen and in profile.log (design 6.6)."""

    def __init__(self, path: Path, stage: str):
        self.path, self.stage = path, stage

    def __call__(self, message: str):
        print(f"[{self.stage}] {message}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{now()} [{self.stage}] {message}\n")


def open_meta(run: "Run") -> dict:
    """Read profile meta.json, or start a new one. Stops when the prep it was made from changed
    (design chapter 10): a new prep version gets a new folder; the same version must not change."""
    prep_hash = file_hash(run.prep_dir / "meta.json")
    path = run.out_dir / "meta.json"
    if not path.exists():
        return {"domain": run.domain_dir.name, "prep_version": run.prep_version, "prep_meta_sha256": prep_hash,
                "config_file": run.config_path.name, "stages": {}}
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("prep_meta_sha256") != prep_hash:
        raise TrackingError(f"prep이 바뀌었습니다: {run.prep_dir / 'meta.json'}의 해시가 {path}의 기록과 다릅니다. "
                            f"새 prep이면 새 버전으로 만들고 그 버전으로 실행하세요")
    return meta


def save_meta(run: "Run", meta: dict):
    write_atomic(run.out_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def check_finished_stage(stage: str, record, config: dict, inputs: dict, output: Path):
    """Design chapter 10: a finished stage is skipped only if it still matches what it was made from."""
    if not record or record.get("status") != "done":
        raise TrackingError(f"{output}이 있지만 meta.json에 {stage} 완료 기록이 없습니다 "
                            f"(실행이 중간에 끊긴 것으로 보입니다). 파일을 지우고 다시 실행하세요")
    problems = []
    for key in sorted(set(config) | set(record.get("config", {}))):
        if record["config"].get(key) != config.get(key):
            problems.append(f"설정 {key}: 기록 {record['config'].get(key)!r}, 지금 {config.get(key)!r}")
    for name in sorted(set(inputs) | set(record.get("inputs", {}))):
        if record["inputs"].get(name) != inputs.get(name):
            problems.append(f"입력 파일 {name}의 해시가 다릅니다")
    if record.get("output") != file_hash(output):
        problems.append(f"출력 파일 {output.name}이 기록 이후 바뀌었습니다")
    if problems:
        raise TrackingError(f"{stage}이 이미 있지만 지금과 맞지 않습니다:\n  - " + "\n  - ".join(problems) +
                            "\n설정을 바꾸려면 새 이름의 설정 파일을 만들고, 다시 만들려면 출력 파일을 직접 지우세요")


def item_raw_stats(prep: Prep, texts: list, review_cap: int) -> dict:
    used = [min(len(review_units(prep, i)), review_cap) for i in range(prep.n_items)]
    return {
        "items_without_reviews": sum(1 for n in used if n == 0),
        "reviews_used": sum(used),
        "reviews_per_item": percentiles(used),
        "words": percentiles([len(t.split()) for t in texts]),
    }


def run_item_raw(run: "Run"):
    """Stage item_raw: prep -> raw/item_raw_profiles.jsonl, recorded in meta.json."""
    stage, output = "item_raw", run.out_dir / ITEM_RAW_FILE
    log = Log(run.out_dir / "profile.log", stage)
    meta = open_meta(run)
    config = {k: run.settings.as_dict()[k] for k in STAGE_CONFIG_KEYS[stage]}
    inputs = {f"prep/{run.prep_version}/{name}": file_hash(run.prep_dir / name) for name in ITEM_RAW_INPUTS}

    if output.exists():
        check_finished_stage(stage, meta["stages"].get(stage), config, inputs, output)
        log(f"이미 완료되어 건너뜁니다: {output}")
        return

    record = {"status": "incomplete", "started": now(), "finished": None, "config": config, "inputs": inputs,
              "env": git_info()}
    meta["stages"][stage] = record
    save_meta(run, meta)
    log(f"시작: {run.prep_dir} -> {output}")

    prep = load_prep(run.prep_dir)
    texts = build_item_raw_profiles(prep, run.settings.seed, run.settings.item_review_cap)
    write_atomic(output, "".join(
        json.dumps({"item_idx": i, "item_id": prep.item_ids[i], "text": t}, ensure_ascii=False) + "\n"
        for i, t in enumerate(texts)))

    stats = item_raw_stats(prep, texts, run.settings.item_review_cap)
    record.update(status="done", finished=now(), count=len(texts), output=file_hash(output), stats=stats)
    save_meta(run, meta)
    log(f"완료: {len(texts):,}개, 리뷰 없는 아이템 {stats['items_without_reviews']:,}개, "
        f"단어 수 중앙값 {stats['words']['median']:,} / 최대 {stats['words']['max']:,}")


STAGE_RUNNERS["item_raw"] = run_item_raw


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
    except TrackingError as e:
        print(f"추적 오류: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
