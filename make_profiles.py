"""Build profiles from a prep version: prep/<version>/ -> profile/<version>_<method>/ (LLM profile design v1.0).

    python make_profiles.py data/<domain> --prep v1 --config configs/profile/<method>.json --stage <stage>

    <stage> = item_raw | item_llm | aux_raw | aux_llm | user_raw | user_llm | embed | all

The command line takes only paths and the stage; every other setting lives in the config file
(design chapter 9). The output folder name is <prep version>_<config file name> (design 8.1).

Implemented stages: none yet (item_raw is being added).

Standard library only (Python 3.10+) for the raw-profile stages.
"""

import argparse
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
    """Exit codes: 0 success, 2 config error, 3 output / tracking error."""
    a = parse_args(argv)
    try:
        run = Run(domain_dir=a.domain_dir, prep_version=a.prep, config_path=a.config, stage=a.stage,
                  settings=load_settings(a.config))
        check_run(run)
    except ValueError as e:
        print(f"설정 오류: {e}")
        return 2
    STAGE_RUNNERS[run.stage](run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
