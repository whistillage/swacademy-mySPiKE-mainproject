import json
from pathlib import Path

import pytest

import validate_dataset as vd

ROOT = Path(__file__).resolve().parent.parent

# A tiny dataset that satisfies every required condition of the spec.
VALID = {
    "interactions.jsonl": [
        {"user_id": "U1", "item_id": "I1", "review": "Great book."},
        {"user_id": "U1", "item_id": "I2", "review": "Nice."},
        {"user_id": "U2", "item_id": "I3", "review": "OK."},
    ],
    "items.csv": "item_id,name\nI1,Book One\nI2,Book Two\nI3,Book Three\n",
    "auxs.csv": "aux_id,name,aux_type\nA1,Author One,person\nA2,Genre Two,genre\n",
    "relations.csv": (
        "relation,head_type,tail_type,sentence,reverse_sentence\n"
        "author,item,aux,{head} lists {tails} as its author,{tail} is credited as the author of {head}\n"
        "sequel,item,item,{head} is followed by {tails},{tail} is the sequel to {head}\n"
    ),
    "kg.csv": (
        "head_type,head,relation,tail_type,tail\n"
        "item,I1,author,aux,A1\nitem,I2,author,aux,A1\nitem,I3,author,aux,A2\nitem,I1,sequel,item,I2\n"
    ),
}


def write_raw(path: Path, files: dict):
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if content is None:
            continue
        if isinstance(content, list):
            content = "".join((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n"
                              for r in content)
        if isinstance(content, bytes):
            (path / name).write_bytes(content)
        else:
            (path / name).write_text(content, encoding="utf-8", newline="")
    return path


def run(tmp_path, **changes):
    """Validate VALID with some files replaced (None = file removed)."""
    files = dict(VALID)
    for key, value in changes.items():
        files[key.replace("__", ".")] = value
    report, data = vd.validate(write_raw(tmp_path / "raw", files))
    return report, data


def inter(*rows):
    return list(VALID["interactions.jsonl"]) + list(rows)


# ---------------------------------------------------------------- valid input

def test_valid_dataset_has_no_issues(tmp_path):
    report, data = run(tmp_path)
    assert report.issues == {}
    assert len(data.interactions) == 3 and len(data.items) == 3 and data.optional_keys == ()


def test_cli_exit_codes_and_json(tmp_path, capsys):
    raw = write_raw(tmp_path / "ok", VALID)
    assert vd.main([str(raw), "--json", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["ok"] is True
    bad = write_raw(tmp_path / "bad", {**VALID, "items.csv": None})
    assert vd.main([str(bad)]) == 1
    assert "E_FILE_MISSING" in capsys.readouterr().out


# ---------------------------------------------------------------- common

def test_missing_file(tmp_path):
    report, data = run(tmp_path, items__csv=None)
    assert "E_FILE_MISSING" in report.codes()
    assert data.items is None
    assert "I_SKIPPED" in report.codes()          # item_id reference check skipped


def test_not_utf8(tmp_path):
    report, _ = run(tmp_path, auxs__csv="aux_id,name,aux_type\nA1,Caf\xe9,place\n".encode("latin-1"))
    assert "E_ENCODING" in report.codes()


def test_bom_is_accepted(tmp_path):
    report, _ = run(tmp_path, items__csv="﻿" + VALID["items.csv"])
    assert report.issues == {}


def test_missing_column(tmp_path):
    report, _ = run(tmp_path, auxs__csv="aux_id,name\nA1,Author One\n")
    assert "E_COLUMNS" in report.codes()


def test_csv_row_with_wrong_field_count(tmp_path):
    report, _ = run(tmp_path, items__csv="item_id,name\nI1,Book, One\nI2,Book Two\nI3,Book Three\n")
    assert "E_CSV_ROW" in report.codes()


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_csv_value(tmp_path, value):
    report, _ = run(tmp_path, items__csv=f"item_id,name\nI1,{value}\nI2,Book Two\nI3,Book Three\n")
    assert "E_EMPTY" in report.codes()


def test_id_whitespace_is_stripped_and_warned(tmp_path):
    report, data = run(tmp_path, items__csv="item_id,name\n I1 ,Book One\nI2,Book Two\nI3,Book Three\n")
    assert report.codes() == {"W_ID_WHITESPACE"}
    assert data.items[0]["item_id"] == "I1"


# ---------------------------------------------------------------- interactions

@pytest.mark.parametrize("line, code", [
    ("{not json", "E_JSON"),
    ('["U1", "I1", "x"]', "E_JSON_OBJECT"),
    ({"user_id": "U3", "item_id": "I1"}, "E_MISSING_KEY"),
    ({"user_id": 123, "item_id": "I1", "review": "x"}, "E_NOT_STRING"),
    ({"user_id": "U3", "item_id": "I1", "review": "   "}, "E_EMPTY"),
    ({"user_id": "U3", "item_id": "NOPE", "review": "x"}, "E_ITEM_REF"),
])
def test_interaction_errors(tmp_path, line, code):
    report, _ = run(tmp_path, interactions__jsonl=inter(line))
    assert code in report.codes(vd.ERROR)


def test_blank_line_is_warned(tmp_path):
    report, data = run(tmp_path, interactions__jsonl=inter("", VALID["interactions.jsonl"][0]))
    assert report.codes() == {"W_BLANK_LINE"}
    assert len(data.interactions) == 4


def test_interaction_id_whitespace(tmp_path):
    report, data = run(tmp_path, interactions__jsonl=inter({"user_id": "U9", "item_id": " I1", "review": "x"}))
    assert report.codes() == {"W_ID_WHITESPACE"}           # stripped, so the item reference is fine
    assert data.interactions[-1]["item_id"] == "I1"


def with_optional(**values):
    """VALID interactions, each with the given optional keys (value list per key)."""
    rows = []
    for i, r in enumerate(VALID["interactions.jsonl"]):
        rows.append({**r, **{k: v[i] for k, v in values.items()}})
    return rows


def test_optional_columns_valid(tmp_path):
    report, data = run(tmp_path, interactions__jsonl=with_optional(
        timestamp=[1167609600, None, 0], rating=[4, 4.111, None]))
    assert report.issues == {}
    assert data.optional_keys == ("timestamp", "rating")


def test_optional_key_must_be_in_every_row(tmp_path):
    rows = with_optional(rating=[5.0, 4.0, 3.0])
    del rows[1]["rating"]
    report, _ = run(tmp_path, interactions__jsonl=rows)
    assert report.codes(vd.ERROR) == {"E_OPTIONAL_KEY"}


@pytest.mark.parametrize("value", [1167609600000, 1.5, "1167609600", True, -1])
def test_bad_timestamp(tmp_path, value):
    report, _ = run(tmp_path, interactions__jsonl=with_optional(timestamp=[value, 1, 2]))
    assert report.codes(vd.ERROR) == {"E_TIMESTAMP"}


@pytest.mark.parametrize("value", [0.5, 5.5, 4.1111, "4.0", True])
def test_bad_rating(tmp_path, value):
    report, _ = run(tmp_path, interactions__jsonl=with_optional(rating=[value, 1.0, 5.0]))
    assert report.codes(vd.ERROR) == {"E_RATING"}


def test_nan_rating(tmp_path):
    rows = ['{"user_id": "U1", "item_id": "I1", "review": "x", "rating": NaN}']
    report, _ = run(tmp_path, interactions__jsonl=rows)
    assert "E_RATING" in report.codes()


# ---------------------------------------------------------------- items / auxs

def test_duplicate_item_id(tmp_path):
    report, _ = run(tmp_path, items__csv=VALID["items.csv"] + "I2,Again\n")
    assert report.codes(vd.ERROR) == {"E_DUP_ITEM"}


def test_duplicate_after_strip(tmp_path):
    report, _ = run(tmp_path, items__csv=VALID["items.csv"] + "I2 ,Again\n")
    assert "E_DUP_ITEM" in report.codes()


def test_duplicate_aux_id(tmp_path):
    report, _ = run(tmp_path, auxs__csv=VALID["auxs.csv"] + "A1,Someone,person\n")
    assert report.codes(vd.ERROR) == {"E_DUP_AUX"}


# ---------------------------------------------------------------- relations

REL_HEADER = "relation,head_type,tail_type,sentence,reverse_sentence\n"
REL_OK = VALID["relations.csv"]


@pytest.mark.parametrize("extra, code", [
    ("author,item,aux,{head} by {tails},{tail} wrote {head}\n", "E_DUP_RELATION"),
    ("nat,aux,place,{head} is from {tails},{tail} is home of {head}\n", "E_REL_TYPE"),
    ("wrote,aux,item,{head} wrote {tails},{tail} was written by {head}\n", "E_REL_AUX_ITEM"),
    ("x,item,aux,{head} has {tail},{tail} of {head}\n", "E_PLACEHOLDER_MISSING"),
    ("x,item,aux,{head} has {tails},{tail} of something\n", "E_PLACEHOLDER_MISSING"),
    ("x,item,aux,{head} has {tails} by {name},{tail} of {head}\n", "E_PLACEHOLDER_UNKNOWN"),
    ("x,item,aux,{head} has {tails} {,{tail} of {head}\n", "E_PLACEHOLDER_UNKNOWN"),
])
def test_relation_errors(tmp_path, extra, code):
    report, _ = run(tmp_path, relations__csv=REL_OK + extra)
    assert code in report.codes(vd.ERROR)


# ---------------------------------------------------------------- real data

@pytest.mark.parametrize("domain", ["book", "book_noisy"])
def test_real_data_has_no_format_errors(domain):
    raw = ROOT / "data" / domain / "raw"
    if not (raw / "interactions.jsonl").exists():
        pytest.skip(f"{raw} not generated")
    report, data = vd.validate(raw)
    assert report.by_level(vd.ERROR) == []
    assert len(data.items) == {"book": 8_696, "book_noisy": 8_746}[domain]
