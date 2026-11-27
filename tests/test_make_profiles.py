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


# ---------------------------------------------------------------- prep input

import preprocess as pp
from dataset_fixtures import write_raw

# 3 users x 3-4 items, so user_min 3 keeps everyone and each user has 1 train / 1 valid / 1 test
# (U3 has 4: 2 train). U1 reviewed I3 twice (duplicate rows merge into one pair).
# With seed 2020 the split is: train U1-I3, U2-I2, U3-I3, U3-I4 (idx (0,2) (1,1) (2,2) (2,3)).
TOY_RAW = {
    "interactions.jsonl": [
        {"user_id": "U1", "item_id": "I1", "review": "u1 on i1"},
        {"user_id": "U1", "item_id": "I2", "review": "u1 on i2"},
        {"user_id": "U1", "item_id": "I3", "review": "u1 on i3 first"},
        {"user_id": "U1", "item_id": "I3", "review": "u1 on i3 second"},
        {"user_id": "U2", "item_id": "I1", "review": "u2 on i1"},
        {"user_id": "U2", "item_id": "I2", "review": "u2 on i2"},
        {"user_id": "U2", "item_id": "I4", "review": "u2 on i4"},
        {"user_id": "U3", "item_id": "I1", "review": "u3 on i1"},
        {"user_id": "U3", "item_id": "I2", "review": "u3 on i2"},
        {"user_id": "U3", "item_id": "I3", "review": "u3 on i3"},
        {"user_id": "U3", "item_id": "I4", "review": "u3 on i4"},
    ],
    "items.csv": "item_id,name\nI1,Book One\nI2,Book Two\nI3,Book Three\nI4,Book Four\n",
    "auxs.csv": "aux_id,name,aux_type\nA1,Author One,person\nA2,Genre Two,genre\n",
    "relations.csv": (
        "relation,head_type,tail_type,sentence,reverse_sentence\n"
        "author,item,aux,{head} lists {tails} as its author,{tail} is credited as the author of {head}\n"
        "sequel,item,item,{head} is followed by {tails},{tail} is the sequel to {head}\n"
    ),
    "kg.csv": (
        "head_type,head,relation,tail_type,tail\n"
        "item,I1,author,aux,A1\nitem,I2,author,aux,A1\nitem,I3,author,aux,A2\nitem,I4,author,aux,A2\n"
        "item,I1,sequel,item,I2\n"
    ),
}


@pytest.fixture
def toy_prep(tmp_path) -> Path:
    """prep/v1 made by the real preprocess.py, so both scripts agree on the file formats."""
    domain = tmp_path / "toy"
    write_raw(domain / "raw", TOY_RAW)
    pp.preprocess(pp.Config(domain_dir=domain, version="v1", user_min_interactions=3))
    return domain / "prep" / "v1"


def read_train(prep_dir: Path) -> set:
    pairs = set()
    for line in (prep_dir / "train.txt").read_text(encoding="utf-8").splitlines():
        u, *its = map(int, line.split())
        pairs.update((u, i) for i in its)
    return pairs


def test_load_prep(toy_prep):
    p = mp.load_prep(toy_prep)
    assert p.item_ids == ["I1", "I2", "I3", "I4"]
    assert p.item_names == ["Book One", "Book Two", "Book Three", "Book Four"]
    assert p.aux_names == ["Author One", "Genre Two"]
    assert p.user_ids == ["U1", "U2", "U3"]
    assert [r.name for r in p.relations] == ["author", "sequel"]
    assert p.relations[1] == mp.Relation("sequel", "item", "item", "{head} is followed by {tails}",
                                         "{tail} is the sequel to {head}")
    # node_idx: items 0-3, aux 4-5
    assert p.n_items == 4 and p.is_item(3) and not p.is_item(4)
    assert p.node_name(0) == "Book One" and p.node_name(5) == "Genre Two"
    assert sorted(p.kg) == [(0, 0, 4), (0, 1, 1), (1, 0, 4), (2, 0, 5), (3, 0, 5)]


def test_load_prep_keeps_train_reviews_only(toy_prep):
    assert read_train(toy_prep) == {(0, 2), (1, 1), (2, 2), (2, 3)}     # the split the test relies on
    p = mp.load_prep(toy_prep)
    assert p.item_reviews == {
        1: [(1, ["u2 on i2"])],
        2: [(0, ["u1 on i3 first", "u1 on i3 second"]), (2, ["u3 on i3"])],   # merged, in input order
        3: [(2, ["u3 on i4"])],
    }                                                                    # I1: no train review at all


@pytest.mark.parametrize("name", ["item_list.csv", "aux_list.csv", "user_list.csv", "relation_list.csv",
                                  "kg.csv", "train.txt", "interactions.jsonl", "meta.json"])
def test_load_prep_missing_file(toy_prep, name):
    (toy_prep / name).unlink()
    with pytest.raises(mp.PrepError, match="prep 파일이 없습니다"):
        mp.load_prep(toy_prep)


def edit(path: Path, fn):
    path.write_text(fn(path.read_text(encoding="utf-8")), encoding="utf-8", newline="")


def test_load_prep_count_mismatch(toy_prep):
    meta = json.loads((toy_prep / "meta.json").read_text(encoding="utf-8"))
    meta["n_items"] = 5
    (toy_prep / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(mp.PrepError, match="item_list.csv: 행 수"):
        mp.load_prep(toy_prep)


def test_load_prep_idx_gap(toy_prep):
    edit(toy_prep / "user_list.csv", lambda s: s.replace("U2,1", "U2,7"))
    with pytest.raises(mp.PrepError, match="user_list.csv: 3행의 user_idx"):
        mp.load_prep(toy_prep)


def test_load_prep_node_idx_rule(toy_prep):
    edit(toy_prep / "aux_list.csv", lambda s: s.replace("A1,0,4", "A1,0,9"))
    with pytest.raises(mp.PrepError, match="node_idx"):
        mp.load_prep(toy_prep)


def test_load_prep_kg_out_of_range(toy_prep):
    edit(toy_prep / "kg.csv", lambda s: s + "0,0,6\n")
    with pytest.raises(mp.PrepError, match="kg.csv: 범위"):
        mp.load_prep(toy_prep)


def test_load_prep_train_out_of_range(toy_prep):
    edit(toy_prep / "train.txt", lambda s: s + "0 4\n")
    with pytest.raises(mp.PrepError, match="train.txt: 범위"):
        mp.load_prep(toy_prep)


def test_load_prep_train_pair_missing_from_interactions(toy_prep):
    u, i = sorted(read_train(toy_prep))[0]
    edit(toy_prep / "interactions.jsonl",
         lambda s: "".join(l for l in s.splitlines(keepends=True)
                           if not l.startswith(f'{{"user_idx": {u}, "item_idx": {i},')))
    with pytest.raises(mp.PrepError, match="train 쌍 1개가 없습니다"):
        mp.load_prep(toy_prep)


def test_main_prep_error_exit_code(toy_prep, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(mp.STAGE_RUNNERS, "item_raw", lambda run: mp.load_prep(run.prep_dir))
    (toy_prep / "kg.csv").unlink()
    config = write_config(tmp_path / "qwen3-4b.json", GOOD)
    assert mp.main([str(toy_prep.parent.parent), "--prep", "v1", "--config", str(config),
                    "--stage", "item_raw"]) == 3
    assert "prep 오류" in capsys.readouterr().out
