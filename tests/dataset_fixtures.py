"""Shared test data: a tiny dataset that satisfies every required condition of the spec."""

import json
from pathlib import Path

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
