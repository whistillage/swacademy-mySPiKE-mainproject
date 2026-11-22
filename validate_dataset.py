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
import statistics
import sys
from collections import Counter, defaultdict
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

# filters (preprocessing design 5.3) and spec 5.6 recommended conditions
USER_MIN_INTERACTIONS = 5
REC_RELATIONS = (3, 30)             # (2) number of used relations
REC_RELATION_MIN_USES = 10          # (3)
REC_MEDIAN_TRIPLES = 5              # (4) median triples per item (as head)
REC_MAX_TRIPLES = 100               # (4) per item
REC_SINGLE_ITEM_AUX = 0.70          # (5) share of aux linked to exactly one item
REC_TOP_RELATION = 0.80             # (6) share of the most used relation
REC_GIANT_AUX = 0.30                # (7) aux linked to this share of all items

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
    "E_KG_TYPE": (ERROR, "head_type/tail_type은 item 또는 aux여야 합니다"),
    "E_KG_REF": (ERROR, "목록에 없는 head/tail/relation입니다 (items.csv, auxs.csv, relations.csv)"),
    "E_KG_AUX_ITEM": (ERROR, "aux → item 트리플은 넣을 수 없습니다 (방향을 item → aux로)"),
    "E_KG_SELF": (ERROR, "자기 자신을 가리키는 트리플입니다"),
    "E_KG_REL_TYPE": (ERROR, "관계를 relations.csv에 선언한 타입 조합과 다르게 썼습니다"),
    "E_KG_REVERSE": (ERROR, "같은 관계의 역방향 트리플이 있습니다 (역방향은 모델이 자동으로 만듭니다)"),
    "E_ITEM_NO_TRIPLE": (ERROR, "kg.csv의 어떤 트리플에도 나오지 않는 아이템입니다"),
    "E_AUX_NOT_LINKED": (ERROR, "아이템과 직접 연결되지 않은 aux입니다 (item → aux 트리플의 tail로 나와야 함)"),
    "W_ID_WHITESPACE": (WARNING, "앞뒤에 공백이 있는 ID가 있습니다 (지운 뒤 처리됩니다)"),
    "W_BLANK_LINE": (WARNING, "빈 줄이 있습니다 (건너뜁니다)"),
    "W_DUP_PAIR": (WARNING, "같은 (user_id, item_id)가 여러 번 있습니다 (전처리에서 하나로 합쳐집니다)"),
    "W_FEW_INTERACTIONS": (WARNING, f"상호작용이 {USER_MIN_INTERACTIONS}건 미만인 사용자입니다 (중복을 합친 뒤 기준, 전처리에서 제외됩니다)"),
    "W_OPTIONAL_ALL_NULL": (WARNING, "권장 컬럼이 있지만 값이 모두 null입니다"),
    "W_ITEM_NO_INTERACTION": (WARNING, "상호작용이 없는 아이템입니다 (전처리에서 제외됩니다)"),
    "W_AUX_TYPE_CASE": (WARNING, "aux_type은 영어 소문자로 통일합니다"),
    "W_AUX_SAME_NAME": (WARNING, "이름과 aux_type이 모두 같은 aux가 여럿입니다 (같은 대상이면 aux_id를 하나로)"),
    "W_TEMPLATE_STYLE": (WARNING, "템플릿에 따옴표나 끝 마침표가 있습니다 (자동으로 붙습니다)"),
    "W_UNUSED_RELATION": (WARNING, "kg.csv에서 쓰이지 않는 관계입니다"),
    "W_KG_DUP_ROW": (WARNING, "완전히 같은 트리플 행이 있습니다 (전처리에서 하나만 남깁니다)"),
    "W_NO_SHARED_AUX": (WARNING, "다른 아이템과 공유하는 aux가 없는 아이템입니다 (권장 5.6-1)"),
    "W_RELATION_COUNT": (WARNING, f"쓰인 관계 수가 권장 범위 {REC_RELATIONS[0]}~{REC_RELATIONS[1]}개 밖입니다 (권장 5.6-2)"),
    "W_RARE_RELATION": (WARNING, f"{REC_RELATION_MIN_USES}회 미만으로 쓰인 관계입니다 (권장 5.6-3)"),
    "W_FEW_TRIPLES": (WARNING, f"아이템당 트리플 수 중앙값이 {REC_MEDIAN_TRIPLES} 미만입니다 (권장 5.6-4)"),
    "W_MANY_TRIPLES": (WARNING, f"트리플이 {REC_MAX_TRIPLES}개를 넘는 아이템입니다 (권장 5.6-4)"),
    "W_SINGLE_ITEM_AUX": (WARNING, f"아이템 하나에만 연결된 aux가 {REC_SINGLE_ITEM_AUX:.0%}를 넘습니다 (권장 5.6-5)"),
    "W_TOP_RELATION": (WARNING, f"한 관계가 전체 트리플의 {REC_TOP_RELATION:.0%}를 넘습니다 (권장 5.6-6)"),
    "W_GIANT_AUX": (WARNING, f"전체 아이템의 {REC_GIANT_AUX:.0%} 이상과 연결된 aux입니다. 유지할지 검토하세요 (권장 5.6-7)"),
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
        self.stats = {}    # file -> {label: value}, statistics for the INFO section

    def add(self, code: str, file: str, example=None, count: int = 1):
        issue = self.issues.setdefault((code, file), Issue(code, file))
        issue.count += count
        if example is not None and len(issue.examples) < self.max_examples:
            issue.examples.append(example)

    def stat(self, file: str, label: str, value):
        self.stats.setdefault(file, {})[label] = value

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
            "stats": self.stats,
        }

    def render(self) -> str:
        lines = []
        for level in (ERROR, WARNING, INFO):
            issues = self.by_level(level)
            lines.append(f"== {level}" if level == INFO else f"== {level} ({len(issues)})")
            for i in issues:
                lines.append(f"[{i.code}] {i.file}: {i.message} ({i.count:,}건)")
                for ex in i.examples:
                    lines.append(f"    - {ex}")
            if level == INFO:
                for file, stats in self.stats.items():
                    lines.append(f"[{file}]")
                    for label, value in stats.items():
                        lines.append(f"    {label}: {format_value(value)}")
            lines.append("")
        lines.append("결과: " + ("에러 없음" if not self.has_errors else "에러 있음 (제출 불가)"))
        return "\n".join(lines)


def format_value(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    if isinstance(v, dict):
        return ", ".join(f"{k} {format_value(x)}" for k, x in v.items())
    return str(v)


def distribution(values) -> dict:
    values = list(values)
    if not values:
        return {"최소": 0, "중앙값": 0, "최대": 0}
    return {"최소": min(values), "중앙값": statistics.median(values), "최대": max(values)}


def share(part: int, whole: int) -> str:
    return f"{part:,} ({part / whole:.1%})" if whole else f"{part:,}"


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
    # filled by the checks (input for the warnings and for preprocessing)
    relation_types: dict = None  # relation -> (head_type, tail_type), valid declarations only
    kg_rows: list = None         # format-valid kg rows as (triple, line), duplicates included
    kg_triples: dict = None      # triple -> first line; deduped, structurally valid
                                 # triple = (head_type, head, relation, tail_type, tail)


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
    data.relation_types = {}
    declared = set()
    for r in data.relations:
        where = f"{name}:{r['_line']} {r['relation']}"
        types = (r["head_type"].strip(), r["tail_type"].strip())
        first = r["relation"] not in declared
        declared.add(r["relation"])
        if any(t not in NODE_TYPES for t in types):
            report.add("E_REL_TYPE", name, f"{where} ({types[0]} → {types[1]})")
        elif types == ("aux", "item"):
            report.add("E_REL_AUX_ITEM", name, where)
        elif first:
            data.relation_types[r["relation"]] = types
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


def check_kg(data: Dataset, report: Report):
    """KG format, then structure on deduped triples. Order matters (one report per cause):
    1. format (E_KG_TYPE, E_KG_REF)            failing rows are dropped; rows using a relation
                                                whose declaration is invalid are dropped silently
    2. dedupe                                   exact duplicates -> one triple (warning in W_*)
    3. per triple, first match only             E_KG_AUX_ITEM > E_KG_SELF > E_KG_REL_TYPE; dropped
    4. pairs (E_KG_REVERSE)                     one report per pair; triples are kept
    Survivors go to data.kg_triples, used by the link checks and the warnings.
    """
    name = "kg.csv"
    if data.kg is None:
        return
    needed = {"items.csv": data.items, "auxs.csv": data.auxs, "relations.csv": data.relations}
    missing = [n for n, rows in needed.items() if rows is None]
    if missing:
        skipped(report, name, "KG 참조·구조 검사", ", ".join(missing))
        return

    nodes = {"item": {r["item_id"] for r in data.items}, "aux": {r["aux_id"] for r in data.auxs}}
    known_relations = {r["relation"] for r in data.relations}

    # 1. format
    data.kg_rows = []
    for r in data.kg:
        where = f"{name}:{r['_line']}"
        ht, tt = r["head_type"].strip(), r["tail_type"].strip()
        if ht not in NODE_TYPES or tt not in NODE_TYPES:
            report.add("E_KG_TYPE", name, f"{where} ({r['head_type']} → {r['tail_type']})")
            continue
        bad = [f"{col}={r[col]!r}" for col, typ in (("head", ht), ("tail", tt)) if r[col] not in nodes[typ]]
        if r["relation"] not in known_relations:
            bad.append(f"relation={r['relation']!r}")
        if bad:
            report.add("E_KG_REF", name, f"{where} " + ", ".join(bad))
            continue
        if r["relation"] not in data.relation_types:
            continue   # invalid declaration, reported for relations.csv
        data.kg_rows.append(((ht, r["head"], r["relation"], tt, r["tail"]), r["_line"]))

    # 2. dedupe
    unique = {}
    for triple, line in data.kg_rows:
        unique.setdefault(triple, line)

    # 3. per-triple structure
    data.kg_triples = {}
    for triple, line in unique.items():
        ht, h, rel, tt, t = triple
        where = f"{name}:{line} {h} --{rel}--> {t}"
        if (ht, tt) == ("aux", "item"):
            report.add("E_KG_AUX_ITEM", name, where)
        elif ht == tt and h == t:
            report.add("E_KG_SELF", name, where)
        elif data.relation_types[rel] != (ht, tt):
            decl = data.relation_types[rel]
            report.add("E_KG_REL_TYPE", name, f"{where} (사용: {ht} → {tt}, 선언: {decl[0]} → {decl[1]})")
        else:
            data.kg_triples[triple] = line

    # 4. reverse pairs (same relation), reported once per pair
    for triple, line in data.kg_triples.items():
        ht, h, rel, tt, t = triple
        rev = (tt, t, rel, ht, h)
        if rev in data.kg_triples and (h, t) < (t, h):
            report.add("E_KG_REVERSE", name,
                       f"{name}:{line} {h} --{rel}--> {t} / {name}:{data.kg_triples[rev]} {t} --{rel}--> {h}")


def check_links(data: Dataset, report: Report):
    """Every item appears in a triple (head or tail); every aux is the tail of an item -> aux triple."""
    if data.kg_triples is None:
        for file in ("items.csv", "auxs.csv"):
            if getattr(data, file.split(".")[0]) is not None:
                skipped(report, file, "KG 등장·연결 검사", "kg.csv (또는 KG 검사에 필요한 파일)")
        return
    in_kg = set()
    linked_aux = set()
    for ht, h, _, tt, t in data.kg_triples:
        if ht == "item":
            in_kg.add(h)
        if tt == "item":
            in_kg.add(t)
        if ht == "item" and tt == "aux":
            linked_aux.add(t)
    for r in data.items:
        if r["item_id"] not in in_kg:
            report.add("E_ITEM_NO_TRIPLE", "items.csv", f"items.csv:{r['_line']} item_id={r['item_id']!r}")
    for r in data.auxs:
        if r["aux_id"] not in linked_aux:
            report.add("E_AUX_NOT_LINKED", "auxs.csv", f"auxs.csv:{r['_line']} aux_id={r['aux_id']!r}")


# ---------------------------------------------------------------- warnings and statistics

def warn_interactions(data: Dataset, report: Report):
    name = "interactions.jsonl"
    rows = data.interactions
    if rows is None:
        return
    lines = defaultdict(list)
    for r in rows:
        lines[(r["user_id"], r["item_id"])].append(r["_line"])
    for (u, i), ls in lines.items():
        if len(ls) > 1:
            report.add("W_DUP_PAIR", name, f"user_id={u!r} item_id={i!r}: {len(ls)}행 (줄 {', '.join(map(str, ls[:5]))})")

    per_user = Counter(u for u, _ in lines)
    few = {u: n for u, n in per_user.items() if n < USER_MIN_INTERACTIONS}
    for u, n in few.items():
        report.add("W_FEW_INTERACTIONS", name, f"user_id={u!r}: {n}건")

    for key in data.optional_keys:
        if all(r.get(key) is None for r in rows):
            report.add("W_OPTIONAL_ALL_NULL", name, key)

    report.stat(name, "행", len(rows))
    report.stat(name, "서로 다른 (user, item) 쌍", len(lines))
    report.stat(name, "중복으로 합쳐질 행", len(rows) - len(lines))
    report.stat(name, "사용자", len(per_user))
    report.stat(name, "사용자당 상호작용", distribution(per_user.values()))
    report.stat(name, f"정확히 {USER_MIN_INTERACTIONS}건인 사용자",
                share(sum(n == USER_MIN_INTERACTIONS for n in per_user.values()), len(per_user)))
    report.stat(name, "필터 예상: 제외될 사용자", share(len(few), len(per_user)))
    report.stat(name, "필터 예상: 제외될 쌍", share(sum(few.values()), len(lines)))
    for key in data.optional_keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        report.stat(name, f"{key} null 비율", share(len(rows) - len(values), len(rows)))
        valid = [v for v in values if (valid_timestamp if key == "timestamp" else valid_rating)(v)]
        if not valid:
            continue
        if key == "timestamp":
            from datetime import datetime, timezone
            day = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
            report.stat(name, "timestamp 기간 (UTC)", f"{day(min(valid))} ~ {day(max(valid))}")
        else:
            report.stat(name, "rating 분포", dict(sorted(Counter(round(v) for v in valid).items())))
            report.stat(name, "rating 평균", statistics.mean(valid))


def warn_items_auxs(data: Dataset, report: Report):
    if data.items is not None:
        report.stat("items.csv", "아이템", len(data.items))
        if data.interactions is None:
            skipped(report, "items.csv", "상호작용 없는 아이템", "interactions.jsonl")
        else:
            used = {r["item_id"] for r in data.interactions}
            for r in data.items:
                if r["item_id"] not in used:
                    report.add("W_ITEM_NO_INTERACTION", "items.csv", f"items.csv:{r['_line']} item_id={r['item_id']!r}")
            report.stat("items.csv", "상호작용 없는 아이템",
                        share(sum(r["item_id"] not in used for r in data.items), len(data.items)))

    if data.auxs is not None:
        name = "auxs.csv"
        groups = defaultdict(list)
        for r in data.auxs:
            t = r["aux_type"].strip()
            if not (t.isascii() and t == t.lower()):
                report.add("W_AUX_TYPE_CASE", name, f"{name}:{r['_line']} aux_type={r['aux_type']!r}")
            groups[(r["name"].strip(), t)].append(r["aux_id"])
        for (n, t), ids in groups.items():
            if len(set(ids)) > 1:
                report.add("W_AUX_SAME_NAME", name, f"{n!r} ({t}): " + ", ".join(ids[:5]))
        report.stat(name, "aux", len(data.auxs))
        report.stat(name, "aux_type별", dict(Counter(r["aux_type"].strip() for r in data.auxs).most_common(10)))


def warn_relations(data: Dataset, report: Report):
    name = "relations.csv"
    if data.relations is None:
        return
    for r in data.relations:
        for col in ("sentence", "reverse_sentence"):
            t = r[col].strip()
            if "'" in t or '"' in t or t.endswith("."):
                report.add("W_TEMPLATE_STYLE", name, f"{name}:{r['_line']} {r['relation']} {col}: {r[col]!r}")
    if data.kg_triples is not None:
        used = {rel for _, _, rel, _, _ in data.kg_triples}
        for r in data.relations:
            if r["relation"] in data.relation_types and r["relation"] not in used:
                report.add("W_UNUSED_RELATION", name, f"{name}:{r['_line']} {r['relation']}")
    report.stat(name, "관계", len(data.relations))
    if data.relation_types:
        report.stat(name, "타입 조합별",
                    dict(Counter(f"{h} → {t}" for h, t in data.relation_types.values()).most_common()))


def warn_kg(data: Dataset, report: Report):
    """Duplicate rows (format-valid rows) and spec 5.6 recommended conditions (structurally valid triples)."""
    name = "kg.csv"
    if data.kg_rows is None:
        return
    first = {}
    for triple, line in data.kg_rows:
        if triple in first:
            report.add("W_KG_DUP_ROW", name, f"{name}:{line} (처음: {first[triple]}행) {triple[1]} --{triple[2]}--> {triple[4]}")
        else:
            first[triple] = line

    triples = data.kg_triples
    items = [r["item_id"] for r in data.items]
    heads = Counter(h for ht, h, _, _, _ in triples if ht == "item")
    aux_items = defaultdict(set)                      # aux -> items linking it (item -> aux)
    item_auxs = defaultdict(set)
    for ht, h, _, tt, t in triples:
        if ht == "item" and tt == "aux":
            aux_items[t].add(h)
            item_auxs[h].add(t)
    rel_uses = Counter(rel for _, _, rel, _, _ in triples)

    # (1) no shared aux
    for item in items:
        if not any(len(aux_items[a]) > 1 for a in item_auxs.get(item, ())):
            report.add("W_NO_SHARED_AUX", name, f"item_id={item!r}")
    # (2) number of used relations
    if not REC_RELATIONS[0] <= len(rel_uses) <= REC_RELATIONS[1]:
        report.add("W_RELATION_COUNT", name, f"{len(rel_uses)}개")
    # (3) rarely used relations
    for rel, n in sorted(rel_uses.items(), key=lambda x: (x[1], x[0])):
        if n < REC_RELATION_MIN_USES:
            report.add("W_RARE_RELATION", name, f"{rel}: {n}회")
    # (4) triples per item (as head; items without outgoing triples count as 0)
    per_item = [heads.get(i, 0) for i in items]
    if per_item and statistics.median(per_item) < REC_MEDIAN_TRIPLES:
        report.add("W_FEW_TRIPLES", name, f"중앙값 {format_value(statistics.median(per_item))}")
    for i in items:
        if heads.get(i, 0) > REC_MAX_TRIPLES:
            report.add("W_MANY_TRIPLES", name, f"item_id={i!r}: {heads[i]}개")
    # (5) aux linked to a single item
    degrees = [len(v) for v in aux_items.values()]
    single = sum(d == 1 for d in degrees)
    if degrees and single / len(degrees) > REC_SINGLE_ITEM_AUX:
        report.add("W_SINGLE_ITEM_AUX", name, share(single, len(degrees)))
    # (6) dominant relation
    if rel_uses:
        rel, n = rel_uses.most_common(1)[0]
        if n / len(triples) > REC_TOP_RELATION:
            report.add("W_TOP_RELATION", name, f"{rel}: {share(n, len(triples))}")
    # (7) giant aux
    for aux, linked in sorted(aux_items.items(), key=lambda x: -len(x[1])):
        if items and len(linked) >= REC_GIANT_AUX * len(items):
            report.add("W_GIANT_AUX", name, f"aux_id={aux!r}: 아이템 {share(len(linked), len(items))}")

    pairs = Counter((ht, h, tt, t) for ht, h, _, tt, t in triples)
    report.stat(name, "행 (형식이 맞는 행)", len(data.kg_rows))
    report.stat(name, "트리플 (중복 제거, 구조 검사 통과)", len(triples))
    report.stat(name, "유형별", dict(Counter(f"{ht} → {tt}" for ht, _, _, tt, _ in triples).most_common()))
    report.stat(name, "쓰인 관계", len(rel_uses))
    report.stat(name, "관계별 사용", dict(rel_uses.most_common()))
    report.stat(name, "아이템당 트리플 (head 기준)", distribution(per_item))
    report.stat(name, "aux당 연결 아이템", distribution(degrees))
    report.stat(name, "아이템 하나에만 연결된 aux", share(single, len(degrees)))
    report.stat(name, "관계가 여러 개인 (head, tail) 쌍", sum(n > 1 for n in pairs.values()))


def validate(raw_dir, max_examples: int = 5):
    """Run all checks. Returns (Report, Dataset)."""
    raw = Path(raw_dir)
    report = Report(max_examples)
    data = load(raw, report)
    check_interactions(data, report)
    check_unique(data.items, "items.csv", "item_id", "E_DUP_ITEM", report)
    check_unique(data.auxs, "auxs.csv", "aux_id", "E_DUP_AUX", report)
    check_relations(data, report)
    check_kg(data, report)
    check_links(data, report)
    warn_interactions(data, report)
    warn_items_auxs(data, report)
    warn_relations(data, report)
    warn_kg(data, report)
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
