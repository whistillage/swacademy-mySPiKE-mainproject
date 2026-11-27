import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import make_profiles as mp
import preprocess as pp
from dataset_fixtures import write_raw

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


# ---------------------------------------------------------------- sentences and sampling

def small_prep(relations, item_names, aux_names=()) -> mp.Prep:
    return mp.Prep(meta={}, item_ids=[f"I{i}" for i in range(len(item_names))], item_names=list(item_names),
                   aux_names=list(aux_names), user_ids=[], relations=relations, kg=[], item_reviews={})


REL_AUTHOR = mp.Relation("author", "item", "aux", "{head} lists {tails} as its author",
                         "{tail} is credited as the author of {head}")
REL_GENRE = mp.Relation("genre", "item", "aux", "{head} is about {tails}", "{tail} is the subject of {head}")
REL_SEQUEL = mp.Relation("sequel", "item", "item", "{head} is followed by {tails}", "{tail} is the sequel to {head}")


def test_quote_keeps_inner_quotes():
    assert mp.quote("Ender's Game") == "'Ender's Game'"


@pytest.mark.parametrize("names, expected", [
    (["'A'"], "'A'"),
    (["'A'", "'B'"], "'A', and 'B'"),
    (["'A'", "'B'", "'C'"], "'A', 'B', and 'C'"),
])
def test_join_names(names, expected):
    assert mp.join_names(names) == expected


@pytest.mark.parametrize("template, expected", [
    ("{head} lists {tails} as its author", "'W' lists 'M' as its author."),
    ("{head} lists {tails} as its author.", "'W' lists 'M' as its author."),       # no '..'
    ("{head} lists {tails} as its author . ", "'W' lists 'M' as its author."),
    ("{head} lists {tails} {x}", "'W' lists 'M' {x}."),                           # other braces untouched
])
def test_fill(template, expected):
    assert mp.fill(template, head="'W'", tails="'M'") == expected


def test_forward_sentences_group_by_relation():
    # items 0-1, aux 2-5. rel_idx: 0 author, 1 genre
    p = small_prep([REL_AUTHOR, REL_GENRE], ["Wicked", "Son"], ["Oz", "Witchcraft", "Maguire", "Oz"])
    edges = [(1, 3), (0, 4), (1, 5), (1, 2)]          # given in a scrambled order
    assert mp.forward_sentences(p, 0, edges) == [
        "'Wicked' lists 'Maguire' as its author.",      # rel 0 before rel 1
        "'Wicked' is about 'Oz', 'Oz', and 'Witchcraft'.",   # name order; equal names by node_idx
    ]


def test_forward_sentences_equal_names_tie_by_node():
    p = small_prep([REL_GENRE], ["W"], ["Same", "Same"])
    # nodes 1 and 2 share a name; the order must not depend on the input order
    assert mp.forward_sentences(p, 0, [(0, 2), (0, 1)]) == mp.forward_sentences(p, 0, [(0, 1), (0, 2)])


def test_forward_sentences_empty():
    assert mp.forward_sentences(small_prep([REL_AUTHOR], ["W"]), 0, []) == []


def test_reverse_sentences_one_per_triple():
    # items 0-2. rel_idx: 0 author (unused), 1 sequel. Item 0 is the sequel to items 2 and 1.
    p = small_prep([REL_AUTHOR, REL_SEQUEL], ["Circus", "Breakheart Pass", "Alpha"])
    assert mp.reverse_sentences(p, 0, [(1, 1), (2, 1)]) == [
        "'Circus' is the sequel to 'Alpha'.",
        "'Circus' is the sequel to 'Breakheart Pass'.",
    ]


def test_entity_rng_is_per_entity_and_kind():
    a = mp.entity_rng(2020, "item", "X").random()
    assert a == mp.entity_rng(2020, "item", "X").random()
    assert a != mp.entity_rng(2020, "user", "X").random()
    assert a != mp.entity_rng(2020, "item", "Y").random()
    assert a != mp.entity_rng(2021, "item", "X").random()


def test_pick_pinned_values():
    """Pins the exact selection: a change here changes every profile made so far."""
    assert mp.pick(list(range(10)), 3, mp.entity_rng(2020, "item", "000100039X")) == [3, 4, 9]
    assert mp.pick(list(range(10)), 3, mp.entity_rng(2020, "user", "000100039X")) == [7, 0, 3]


def test_pick_independent_of_hash_seed():
    code = ("import make_profiles as mp; "
            "print(mp.pick(list(range(100)), 5, mp.entity_rng(2020, 'item', 'abc')))")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": h}).stdout for h in ("0", "1", "12345")}
    assert len(outs) == 1


def test_pick_all_when_few():
    rng = mp.entity_rng(2020, "item", "X")
    assert mp.pick(["c", "a", "b"], 3, rng) == ["c", "a", "b"]   # input order kept
    assert mp.pick([], 5, rng) == []


def test_pick_sample_when_many():
    got = mp.pick(list("abcdefgh"), 5, mp.entity_rng(2020, "item", "X"))
    assert len(got) == 5 and len(set(got)) == 5 and set(got) <= set("abcdefgh")
    assert mp.pick(list("abc"), 0, mp.entity_rng(2020, "item", "X")) == []


# ---------------------------------------------------------------- item raw profiles

def test_build_item_raw_profiles(toy_prep):
    p = mp.load_prep(toy_prep)
    assert mp.build_item_raw_profiles(p, seed=2020, review_cap=5) == [
        # no train review: KG sentences only
        "'Book One' lists 'Author One' as its author. 'Book One' is followed by 'Book Two'.",
        # reverse item -> item sentence after the forward ones
        "'Book Two' lists 'Author One' as its author. 'Book Two' is the sequel to 'Book One'. "
        "Some users say: 'u2 on i2'",
        # every review is a unit (U1's two merged reviews too), sorted by user_id then list order
        "'Book Three' lists 'Genre Two' as its author. "
        "Some users say: 'u1 on i3 first', 'u1 on i3 second', 'u3 on i3'",
        "'Book Four' lists 'Genre Two' as its author. Some users say: 'u3 on i4'",
    ]


def test_item_raw_review_cap(toy_prep):
    p = mp.load_prep(toy_prep)
    capped = mp.build_item_raw_profiles(p, seed=2020, review_cap=2)
    got = capped[2].split("Some users say: ")[1]
    assert got.count("', '") == 1                     # two reviews
    assert capped == mp.build_item_raw_profiles(p, seed=2020, review_cap=2)   # same every time
    none = mp.build_item_raw_profiles(p, seed=2020, review_cap=0)
    assert all("Some users say" not in t for t in none)


def test_item_raw_review_order_follows_user_id_and_selection_is_per_item(toy_prep):
    p = mp.load_prep(toy_prep)
    units = mp.review_units(p, 2)
    assert units == ["u1 on i3 first", "u1 on i3 second", "u3 on i3"]
    # the selection uses the item's own generator, seeded by its original ID
    expected = mp.pick(units, 2, mp.entity_rng(2020, "item", "I3"))
    assert mp.build_item_raw_profiles(p, 2020, 2)[2].endswith(", ".join(f"'{r}'" for r in expected))


def test_item_raw_quotes_in_reviews_kept():
    p = small_prep([REL_AUTHOR], ["W"], ["M"])
    p.user_ids = ["U1"]
    p.item_reviews = {0: [(0, ["it's 'great'"])]}
    assert mp.item_raw_profile(p, 0, [(0, 1)], [], 2020, 5) == \
        "'W' lists 'M' as its author. Some users say: 'it's 'great''"


def test_item_raw_only_as_item_tail():
    """An item that is only the tail of item -> item triples gets reverse sentences only."""
    p = small_prep([REL_SEQUEL], ["Circus", "Breakheart Pass"])
    p.kg = [(1, 0, 0)]                               # Breakheart Pass is followed by Circus
    assert mp.build_item_raw_profiles(p, 2020, 5) == [
        "'Circus' is the sequel to 'Breakheart Pass'.",
        "'Breakheart Pass' is followed by 'Circus'.",
    ]


def test_item_raw_empty_profile_is_prep_error():
    p = small_prep([REL_AUTHOR], ["W"])
    with pytest.raises(mp.PrepError, match="I0"):
        mp.build_item_raw_profiles(p, 2020, 5)


# ---------------------------------------------------------------- output and tracking

def run_cli(domain: Path, config: Path, stage="item_raw") -> int:
    return mp.main([str(domain), "--prep", "v1", "--config", str(config), "--stage", stage])


@pytest.fixture
def toy_run(toy_prep, tmp_path):
    """(domain folder, config path, profile folder) for a CLI run on the toy prep."""
    config = write_config(tmp_path / "configs" / "qwen3-4b.json", GOOD)
    domain = toy_prep.parent.parent
    return domain, config, domain / "profile" / "v1_qwen3-4b"


def read_meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def test_item_raw_cli_writes_profiles_and_meta(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    lines = (out / "raw" / "item_raw_profiles.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines]
    expected = mp.build_item_raw_profiles(mp.load_prep(domain / "prep" / "v1"), 2020, 5)
    assert rows == [{"item_idx": i, "item_id": f"I{i + 1}", "text": t} for i, t in enumerate(expected)]

    meta = read_meta(out)
    assert meta["domain"] == "toy" and meta["prep_version"] == "v1" and meta["config_file"] == "qwen3-4b.json"
    assert meta["prep_meta_sha256"] == mp.file_hash(domain / "prep" / "v1" / "meta.json")
    rec = meta["stages"]["item_raw"]
    assert rec["status"] == "done" and rec["count"] == 4 and rec["started"] and rec["finished"]
    assert rec["config"] == {"seed": 2020, "item_review_cap": 5}
    assert sorted(rec["inputs"]) == sorted(f"prep/v1/{n}" for n in mp.ITEM_RAW_INPUTS)
    assert rec["output"] == mp.file_hash(out / "raw" / "item_raw_profiles.jsonl")
    assert set(rec["env"]) == {"git_commit", "git_dirty"}
    assert rec["stats"]["items_without_reviews"] == 1
    assert rec["stats"]["reviews_used"] == 5                     # 0 + 1 + 3 + 1
    assert rec["stats"]["reviews_per_item"] == {"median": 1, "p90": 3, "p99": 3, "max": 3}
    assert rec["stats"]["words"]["max"] == max(len(t.split()) for t in expected)

    log = (out / "profile.log").read_text(encoding="utf-8")
    assert "[item_raw] 시작" in log and "[item_raw] 완료: 4개" in log
    assert not list(out.rglob("*.tmp"))
    assert "완료: 4개" in capsys.readouterr().out


def test_item_raw_rerun_skips(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    before = (out / "raw" / "item_raw_profiles.jsonl").read_bytes(), read_meta(out)
    assert run_cli(domain, config) == 0
    assert "이미 완료되어 건너뜁니다" in capsys.readouterr().out
    assert ((out / "raw" / "item_raw_profiles.jsonl").read_bytes(), read_meta(out)) == before


def test_item_raw_deleted_output_is_rebuilt_identically(toy_run):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    first = (out / "raw" / "item_raw_profiles.jsonl").read_bytes()
    (out / "raw" / "item_raw_profiles.jsonl").unlink()
    assert run_cli(domain, config) == 0
    assert (out / "raw" / "item_raw_profiles.jsonl").read_bytes() == first


def test_item_raw_config_change_stops(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    write_config(config, {**GOOD, "item_review_cap": 3})
    assert run_cli(domain, config) == 3
    assert "설정 item_review_cap: 기록 5, 지금 3" in capsys.readouterr().out


def test_item_raw_prep_change_stops(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    edit(domain / "prep" / "v1" / "meta.json", lambda s: s + "\n")
    assert run_cli(domain, config) == 3
    assert "prep이 바뀌었습니다" in capsys.readouterr().out


def test_item_raw_input_change_stops(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    edit(domain / "prep" / "v1" / "kg.csv", lambda s: s + "\n")
    assert run_cli(domain, config) == 3
    assert "입력 파일 prep/v1/kg.csv의 해시가 다릅니다" in capsys.readouterr().out


def test_item_raw_edited_output_stops(toy_run, capsys):
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    edit(out / "raw" / "item_raw_profiles.jsonl", lambda s: s.replace("Book One", "Book 1"))
    assert run_cli(domain, config) == 3
    assert "기록 이후 바뀌었습니다" in capsys.readouterr().out


def test_item_raw_output_without_done_record_stops(toy_run, capsys):
    """Interrupted between the rename of the output and the final meta.json write."""
    domain, config, out = toy_run
    assert run_cli(domain, config) == 0
    meta = read_meta(out)
    meta["stages"]["item_raw"]["status"] = "incomplete"
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert run_cli(domain, config) == 3
    assert "완료 기록이 없습니다" in capsys.readouterr().out


def test_other_config_name_gets_its_own_folder(toy_run):
    domain, config, out = toy_run
    other = write_config(config.parent / "qwen3-4b-cap3.json", {**GOOD, "item_review_cap": 3})
    assert run_cli(domain, config) == 0 and run_cli(domain, other) == 0
    assert read_meta(domain / "profile" / "v1_qwen3-4b-cap3")["stages"]["item_raw"]["config"]["item_review_cap"] == 3
    assert read_meta(out)["stages"]["item_raw"]["config"]["item_review_cap"] == 5


def test_git_info_without_git(monkeypatch):
    def fail(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(mp.subprocess, "run", fail)
    assert mp.git_info() == {"git_commit": None, "git_dirty": None}


def test_git_info_in_repo():
    info = mp.git_info()
    assert len(info["git_commit"]) == 40 and isinstance(info["git_dirty"], bool)


def test_percentiles():
    assert mp.percentiles([5]) == {"median": 5, "p90": 5, "p99": 5, "max": 5}
    assert mp.percentiles(list(range(1, 101))) == {"median": 50, "p90": 90, "p99": 99, "max": 100}
