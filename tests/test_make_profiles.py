import json
from pathlib import Path

import pytest

import make_profiles as mp

ROOT = Path(__file__).resolve().parent.parent
GOOD = {"seed": 2020, "item_review_cap": 5}


def write_config(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def make_domain(tmp_path: Path, prep="v1") -> Path:
    domain = tmp_path / "data" / "toy"
    (domain / "prep" / prep).mkdir(parents=True)
    (domain / "prep" / prep / "meta.json").write_text("{}", encoding="utf-8")
    return domain


# ---------------------------------------------------------------- config file

def test_repo_config_loads():
    s = mp.load_settings(ROOT / "configs" / "profile" / "qwen3-4b.json")
    assert s == mp.Settings(seed=2020, item_review_cap=5)


def test_settings_as_dict_keeps_key_order():
    assert list(mp.Settings(**GOOD).as_dict()) == list(mp.CONFIG_KEYS)


@pytest.mark.parametrize("data, message", [
    ({**GOOD, "item_reveiw_cap": 5}, "알 수 없는 키"),
    ({"seed": 2020}, "빠진 키"),
    ({**GOOD, "seed": "2020"}, "int"),
    ({**GOOD, "seed": 2020.0}, "int"),
    ({**GOOD, "item_review_cap": True}, "int"),
    ({**GOOD, "item_review_cap": -1}, "0 이상"),
    ([1, 2], "JSON 객체"),
    ("{not json", "올바른 JSON"),
])
def test_bad_config(tmp_path, data, message):
    path = write_config(tmp_path / "bad.json", data)
    with pytest.raises(ValueError, match=message):
        mp.load_settings(path)


def test_missing_config_file(tmp_path):
    with pytest.raises(ValueError, match="설정 파일이 없습니다"):
        mp.load_settings(tmp_path / "nope.json")


def test_zero_review_cap_allowed(tmp_path):
    path = write_config(tmp_path / "c.json", {**GOOD, "item_review_cap": 0})
    assert mp.load_settings(path).item_review_cap == 0


# ---------------------------------------------------------------- run paths and checks

def make_run(domain: Path, config: Path, stage="item_raw", prep="v1") -> mp.Run:
    return mp.Run(domain_dir=domain, prep_version=prep, config_path=config, stage=stage,
                  settings=mp.Settings(**GOOD))


def test_out_dir_from_prep_version_and_config_name(tmp_path):
    run = make_run(tmp_path / "data" / "book", Path("configs/profile/qwen3-4b.json"))
    assert run.method == "qwen3-4b"
    assert run.prep_dir == tmp_path / "data" / "book" / "prep" / "v1"
    assert run.out_dir == tmp_path / "data" / "book" / "profile" / "v1_qwen3-4b"


@pytest.mark.parametrize("name", ["Qwen3-4b.json", "qwen3_4b.json", "qwen 3.json", "qwen3-4b.yaml"])
def test_bad_config_name(tmp_path, monkeypatch, name):
    monkeypatch.setitem(mp.STAGE_RUNNERS, "item_raw", lambda run: None)
    run = make_run(make_domain(tmp_path), tmp_path / name)
    with pytest.raises(ValueError, match="설정 파일"):
        mp.check_run(run)


def test_missing_prep(tmp_path, monkeypatch):
    monkeypatch.setitem(mp.STAGE_RUNNERS, "item_raw", lambda run: None)
    run = make_run(make_domain(tmp_path), tmp_path / "qwen3-4b.json", prep="v2")
    with pytest.raises(ValueError, match="prep 폴더"):
        mp.check_run(run)


def test_unimplemented_stage(tmp_path):
    run = make_run(make_domain(tmp_path), tmp_path / "qwen3-4b.json", stage="item_llm")
    with pytest.raises(ValueError, match="아직 구현되지 않은 단계"):
        mp.check_run(run)


# ---------------------------------------------------------------- command line

def test_main_config_error_exit_code(tmp_path, capsys):
    domain = make_domain(tmp_path)
    config = write_config(tmp_path / "qwen3-4b.json", {"seed": 1})
    assert mp.main([str(domain), "--prep", "v1", "--config", str(config), "--stage", "item_raw"]) == 2
    assert "빠진 키" in capsys.readouterr().out


def test_main_unimplemented_stage_exit_code(tmp_path, capsys):
    domain = make_domain(tmp_path)
    config = write_config(tmp_path / "qwen3-4b.json", GOOD)
    assert mp.main([str(domain), "--prep", "v1", "--config", str(config), "--stage", "embed"]) == 2
    assert "아직 구현되지 않은 단계" in capsys.readouterr().out


def test_main_runs_stage(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(mp.STAGE_RUNNERS, "item_raw", calls.append)
    domain = make_domain(tmp_path)
    config = write_config(tmp_path / "qwen3-4b.json", GOOD)
    assert mp.main([str(domain), "--prep", "v1", "--config", str(config), "--stage", "item_raw"]) == 0
    assert [r.out_dir for r in calls] == [domain / "profile" / "v1_qwen3-4b"]


def test_unknown_stage_rejected_by_argparse():
    with pytest.raises(SystemExit):
        mp.main(["data/x", "--prep", "v1", "--config", "c.json", "--stage", "items"])
