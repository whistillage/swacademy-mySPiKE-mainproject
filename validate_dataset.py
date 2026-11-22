"""Validate a domain dataset submission (data spec v1.4, preprocessing design v1.2 section 4.2).

    python validate_dataset.py data/<domain>/raw [--json report.json] [--examples 5]

Prints ERROR / WARNING / INFO sections. Exit code 0 if there is no ERROR, 1 otherwise.
Preprocessing calls validate() and reuses the loaded data (IDs already stripped).

Standard library only (Python 3.10+).
"""

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FILES = {
    "interactions.jsonl": ("user_id", "item_id", "review"),
    "items.csv": ("item_id", "name"),
    "auxs.csv": ("aux_id", "name", "aux_type"),
    "kg.csv": ("head_type", "head", "relation", "tail_type", "tail"),
    "relations.csv": ("relation", "head_type", "tail_type", "sentence", "reverse_sentence"),
}
ID_FIELDS = {
    "interactions.jsonl": ("user_id", "item_id"),
    "items.csv": ("item_id",),
    "auxs.csv": ("aux_id",),
    "kg.csv": ("head", "relation", "tail"),
    "relations.csv": ("relation",),
}
OPTIONAL_INTERACTION_KEYS = ("timestamp", "rating")
NODE_TYPES = ("item", "aux")
PLACEHOLDERS = ("head", "tail", "tails")
REQUIRED_PLACEHOLDERS = {"sentence": ("head", "tails"), "reverse_sentence": ("tail", "head")}
TIMESTAMP_LIMIT = 10 ** 11          # seconds; 13-digit millisecond values are >= 10**12

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"

# code -> (level, message). Messages are for the data team (Korean).
MESSAGES = {
    "E_FILE_MISSING": (ERROR, "파일이 없습니다"),
    "E_ENCODING": (ERROR, "UTF-8로 읽을 수 없습니다"),
    "E_COLUMNS": (ERROR, "필수 컬럼이 없습니다"),
    "E_CSV_ROW": (ERROR, "열 개수가 헤더와 다른 행이 있습니다 (쉼표가 든 값은 큰따옴표로 감싸야 합니다)"),
    "E_EMPTY": (ERROR, "필수 컬럼에 빈 값이 있습니다 (공백만 있어도 빈 값)"),
    "E_JSON": (ERROR, "JSON으로 읽을 수 없는 줄이 있습니다"),
    "E_JSON_OBJECT": (ERROR, "JSON 객체({...})가 아닌 줄이 있습니다"),
    "E_MISSING_KEY": (ERROR, "필수 키가 없는 줄이 있습니다"),
    "E_NOT_STRING": (ERROR, "문자열이어야 하는 값이 문자열이 아닙니다 (ID는 \"123\"처럼 따옴표로)"),
    "E_ITEM_REF": (ERROR, "items.csv에 없는 item_id입니다"),
    "E_OPTIONAL_KEY": (ERROR, "권장 컬럼이 일부 행에만 있습니다 (넣었다면 모든 행에 두고, 모르는 값은 null)"),
    "E_TIMESTAMP": (ERROR, "timestamp가 초 단위 Unix 시간(정수, 0 이상 10^11 미만) 또는 null이 아닙니다"),
    "E_RATING": (ERROR, "rating이 1.0~5.0 숫자(소수점 아래 셋째 자리까지) 또는 null이 아닙니다"),
    "E_DUP_ITEM": (ERROR, "item_id가 중복됩니다"),
    "E_DUP_AUX": (ERROR, "aux_id가 중복됩니다"),
    "E_DUP_RELATION": (ERROR, "relation이 중복됩니다"),
    "E_REL_TYPE": (ERROR, "head_type/tail_type은 item 또는 aux여야 합니다"),
    "E_REL_AUX_ITEM": (ERROR, "aux → item 관계는 선언할 수 없습니다 (방향을 item → aux로)"),
    "E_PLACEHOLDER_MISSING": (ERROR, "템플릿에 필수 자리표시자가 없습니다 (sentence: {head} {tails}, reverse_sentence: {tail} {head})"),
    "E_PLACEHOLDER_UNKNOWN": (ERROR, "템플릿에 허용되지 않은 중괄호 표현이 있습니다 ({head} {tail} {tails}만 사용)"),
    "W_ID_WHITESPACE": (WARNING, "앞뒤에 공백이 있는 ID가 있습니다 (지운 뒤 처리됩니다)"),
    "W_BLANK_LINE": (WARNING, "빈 줄이 있습니다 (건너뜁니다)"),
    "I_SKIPPED": (INFO, "앞의 에러 때문에 검사를 건너뛰었습니다"),
}


# ---------------------------------------------------------------- report

@dataclass
class Issue:
    code: str
    file: str
    count: int = 0
    examples: list = field(default_factory=list)

    @property
    def level(self):
        return MESSAGES[self.code][0]

    @property
    def message(self):
        return MESSAGES[self.code][1]


class Report:
    def __init__(self, max_examples: int = 5):
        self.max_examples = max_examples
        self.issues = {}   # (code, file) -> Issue, in insertion order

    def add(self, code: str, file: str, example=None, count: int = 1):
        issue = self.issues.setdefault((code, file), Issue(code, file))
        issue.count += count
        if example is not None and len(issue.examples) < self.max_examples:
            issue.examples.append(example)

    def by_level(self, level):
        return [i for i in self.issues.values() if i.level == level]

    @property
    def has_errors(self):
        return bool(self.by_level(ERROR))

    def codes(self, level=None):
        return {i.code for i in self.issues.values() if level is None or i.level == level}

    def to_dict(self):
        return {
            "ok": not self.has_errors,
            "issues": [{"level": i.level, "code": i.code, "file": i.file, "message": i.message,
                        "count": i.count, "examples": i.examples} for i in self.issues.values()],
        }

    def render(self) -> str:
        lines = []
        for level in (ERROR, WARNING, INFO):
            issues = self.by_level(level)
            lines.append(f"== {level} ({len(issues)})")
            for i in issues:
                lines.append(f"[{i.code}] {i.file}: {i.message} ({i.count:,}건)")
                for ex in i.examples:
                    lines.append(f"    - {ex}")
            lines.append("")
        lines.append("결과: " + ("에러 없음" if not self.has_errors else "에러 있음 (제출 불가)"))
        return "\n".join(lines)


# ---------------------------------------------------------------- loading

@dataclass
class Dataset:
    """Loaded rows with stripped IDs. A file that could not be read is None."""
    interactions: list = None   # dicts; line number under "_line"
    items: list = None          # dicts; CSV line number under "_line"
    auxs: list = None
    kg: list = None
    relations: list = None
    optional_keys: tuple = ()   # optional interaction keys present in the file


def is_empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def strip_ids(row: dict, name: str, where: str, report: Report):
    for key in ID_FIELDS[name]:
        value = row.get(key)
        if isinstance(value, str) and value != value.strip():
            report.add("W_ID_WHITESPACE", name, f"{where} {key}={value!r}")
            row[key] = value.strip()


def load_csv(raw: Path, name: str, report: Report):
    path = raw / name
    if not path.exists():
        report.add("E_FILE_MISSING", name)
        return None
    try:
        # utf-8-sig: a BOM (e.g. from Excel) is not part of the first column name
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = [c for c in FILES[name] if c not in header]
            if missing:
                report.add("E_COLUMNS", name, f"없는 컬럼: {', '.join(missing)} (헤더: {', '.join(header)})")
                return None
            rows = []
            for row in reader:
                where = f"{name}:{reader.line_num}"
                if None in row or any(v is None for v in row.values()):
                    report.add("E_CSV_ROW", name, where)
                    row.pop(None, None)
                empty = [c for c in FILES[name] if is_empty(row.get(c))]
                if empty:
                    report.add("E_EMPTY", name, f"{where} 빈 컬럼: {', '.join(empty)}")
                    continue
                strip_ids(row, name, where, report)
                row["_line"] = reader.line_num
                rows.append(row)
    except UnicodeDecodeError as e:
        report.add("E_ENCODING", name, str(e))
        return None
    return rows


def load_interactions(raw: Path, report: Report):
    name = "interactions.jsonl"
    path = raw / name
    if not path.exists():
        report.add("E_FILE_MISSING", name)
        return None, ()
    rows = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = f.read().split("\n")
    except UnicodeDecodeError as e:
        report.add("E_ENCODING", name, str(e))
        return None, ()
    if lines and lines[-1] == "":
        lines.pop()   # trailing newline
    for no, text in enumerate(lines, start=1):
        where = f"{name}:{no}"
        if not text.strip():
            report.add("W_BLANK_LINE", name, where)
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            report.add("E_JSON", name, f"{where} {e.msg}")
            continue
        if not isinstance(obj, dict):
            report.add("E_JSON_OBJECT", name, where)
            continue
        missing = [k for k in FILES[name] if k not in obj]
        if missing:
            report.add("E_MISSING_KEY", name, f"{where} 없는 키: {', '.join(missing)}")
            continue
        not_str = [k for k in FILES[name] if not isinstance(obj[k], str)]
        if not_str:
            report.add("E_NOT_STRING", name, f"{where} " + ", ".join(f"{k}={obj[k]!r}" for k in not_str))
            continue
        empty = [k for k in FILES[name] if is_empty(obj[k])]
        if empty:
            report.add("E_EMPTY", name, f"{where} 빈 키: {', '.join(empty)}")
            continue
        strip_ids(obj, name, where, report)
        obj["_line"] = no
        rows.append(obj)
    optional = tuple(k for k in OPTIONAL_INTERACTION_KEYS if any(k in r for r in rows))
    return rows, optional


def load(raw: Path, report: Report) -> Dataset:
    data = Dataset()
    data.interactions, data.optional_keys = load_interactions(raw, report)
    data.items = load_csv(raw, "items.csv", report)
    data.auxs = load_csv(raw, "auxs.csv", report)
    data.kg = load_csv(raw, "kg.csv", report)
    data.relations = load_csv(raw, "relations.csv", report)
    return data


# ---------------------------------------------------------------- checks

def skipped(report: Report, file: str, check: str, needs: str):
    report.add("I_SKIPPED", file, f"{check} (필요한 파일: {needs})")


def valid_timestamp(v) -> bool:
    if v is None:
        return True
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < TIMESTAMP_LIMIT


def valid_rating(v) -> bool:
    if v is None:
        return True
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return False
    return 1.0 <= v <= 5.0 and round(v, 3) == v


def check_interactions(data: Dataset, report: Report):
    name = "interactions.jsonl"
    if data.interactions is None:
        return
    rows = data.interactions

    for key in data.optional_keys:
        for r in rows:
            if key not in r:
                report.add("E_OPTIONAL_KEY", name, f"{name}:{r['_line']} {key} 없음")
    for r in rows:
        if "timestamp" in r and not valid_timestamp(r["timestamp"]):
            report.add("E_TIMESTAMP", name, f"{name}:{r['_line']} timestamp={r['timestamp']!r}")
        if "rating" in r and not valid_rating(r["rating"]):
            report.add("E_RATING", name, f"{name}:{r['_line']} rating={r['rating']!r}")

    if data.items is None:
        skipped(report, name, "item_id 참조", "items.csv")
    else:
        item_ids = {r["item_id"] for r in data.items}
        for r in rows:
            if r["item_id"] not in item_ids:
                report.add("E_ITEM_REF", name, f"{name}:{r['_line']} item_id={r['item_id']!r}")


def check_unique(rows, name: str, key: str, code: str, report: Report):
    if rows is None:
        return
    first = {}
    for r in rows:
        k = r[key]
        if k in first:
            report.add(code, name, f"{name}:{r['_line']} {key}={k!r} (처음: {first[k]}행)")
        else:
            first[k] = r["_line"]


def check_relations(data: Dataset, report: Report):
    name = "relations.csv"
    if data.relations is None:
        return
    check_unique(data.relations, name, "relation", "E_DUP_RELATION", report)
    for r in data.relations:
        where = f"{name}:{r['_line']} {r['relation']}"
        types = (r["head_type"].strip(), r["tail_type"].strip())
        if any(t not in NODE_TYPES for t in types):
            report.add("E_REL_TYPE", name, f"{where} ({types[0]} → {types[1]})")
        elif types == ("aux", "item"):
            report.add("E_REL_AUX_ITEM", name, where)
        for col, required in REQUIRED_PLACEHOLDERS.items():
            template = r[col]
            found = re.findall(r"\{([^{}]*)\}", template)
            missing = [p for p in required if p not in found]
            if missing:
                report.add("E_PLACEHOLDER_MISSING", name,
                           f"{where} {col}: " + ", ".join("{" + p + "}" for p in missing) + " 없음")
            leftover = re.sub(r"\{(?:" + "|".join(PLACEHOLDERS) + r")\}", "", template)
            unknown = [p for p in found if p not in PLACEHOLDERS]
            if unknown or "{" in leftover or "}" in leftover:
                report.add("E_PLACEHOLDER_UNKNOWN", name, f"{where} {col}: {template!r}")


def validate(raw_dir, max_examples: int = 5):
    """Run all checks. Returns (Report, Dataset)."""
    raw = Path(raw_dir)
    report = Report(max_examples)
    data = load(raw, report)
    check_interactions(data, report)
    check_unique(data.items, "items.csv", "item_id", "E_DUP_ITEM", report)
    check_unique(data.auxs, "auxs.csv", "aux_id", "E_DUP_AUX", report)
    check_relations(data, report)
    return report, data


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate a SPiKE domain dataset (raw/ folder).")
    p.add_argument("raw_dir", type=Path, help="folder with the 5 spec files, e.g. data/book/raw")
    p.add_argument("--json", type=Path, help="also write the report as JSON")
    p.add_argument("--examples", type=int, default=5, help="examples shown per issue (default 5)")
    args = p.parse_args(argv)

    report, _ = validate(args.raw_dir, args.examples)
    print(report.render())
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
