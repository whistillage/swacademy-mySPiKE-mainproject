"""Convert the SPiKE authors' amazon-book data (Layer 1) into the 5-file spec format.

    python tools/convert_book.py <author_book_dir> <out_raw_dir> [--force]

Example:
    python tools/convert_book.py ../SPiKE/spike/datasets/kg/amazon-book_kg data/book/raw

Reads (author format, read only):
    user_list.txt, item_list.txt, entity_list.txt, relation_list.txt,
    interaction_list.txt, review_list.txt, kg_final.txt
Writes (data spec v1.2):
    interactions.jsonl, items.csv, auxs.csv, kg.csv, relations.csv

Conversion rules (preprocessing design doc, section 10.1):
- IDs go back to the original strings (user ID, ASIN, DBpedia-style entity ID).
- Item names come from item_list's dbpedia_id: drop "res:", "_" -> " ".
- Aux = entities after the items. name: drop "cat:"/"res:", "_" -> " ".
  aux_type by prefix: cat: -> category, res: -> resource, URL -> url.
- kg_final is kept as is, including its exact duplicate rows (validation warns,
  preprocessing removes them). All triples are item -> aux.
- Empty reviews (5 in the author data) are filled with EMPTY_REVIEW_PLACEHOLDER.
- The author data has no timestamp / rating, so those optional columns are omitted.

Standard library only.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

EMPTY_REVIEW_PLACEHOLDER = "(no review)"

OUTPUT_FILES = ("interactions.jsonl", "items.csv", "auxs.csv", "kg.csv", "relations.csv")

# Relation templates for the book KG.
#   forward: paper appendix Table 8, 1-hop template ([ITEM] -> {head}, [ENTITY] -> {tails},
#            trailing period removed).
#   reverse: back-traced from the authors' entity_raw_profiles.json
#            ("'<tail>' <phrase> '<head>'"), phrase found for all 20 relations.
RELATION_TEMPLATES = {
    "dct:subject": ("{head} is about {tails}", "{tail} is the subject of {head}"),
    "dbo:author": ("{head} lists {tails} as its author", "{tail} is credited as the author of {head}"),
    "dbo:literaryGenre": ("{head} belongs to the literary genre {tails}", "{tail} is the literary genre of {head}"),
    "dbo:subsequentWork": ("{head} is followed by {tails}", "{tail} is the sequel to {head}"),
    "dbo:previousWork": ("{head} comes after {tails}", "{tail} comes before {head}"),
    "dbo:series": ("{head} is part of the series called {tails}", "{tail} is a series that includes {head}"),
    "dbo:director": ("{head} was directed by {tails}", "{tail} directed {head}"),
    "dbo:musicComposer": ("{head} features music composed by {tails}", "{tail} composed the music for {head}"),
    "dbo:producer": ("{head} was produced by {tails}", "{tail} produced {head}"),
    "dbo:starring": ("{head} stars {tails}", "{tail} starred in {head}"),
    "dbo:writer": ("{head} was written by {tails}", "{tail} wrote {head}"),
    "dbo:genre": ("{head} is categorized under the genre {tails}", "{tail} is a genre classification for {head}"),
    "dbo:composer": ("{head} includes compositions by {tails}", "{tail} composed the pieces in {head}"),
    "dbo:creator": ("{head} was created by {tails}", "{tail} created {head}"),
    "dbo:executiveProducer": ("{head} had {tails} as executive producer", "{tail} served as executive producer for {head}"),
    "dbo:notableWork": ("{head} is best known for {tails}", "{tail} is notable for the work {head}"),
    "dbo:award": ("{head} received the award titled {tails}", "{tail} received an award for {head}"),
    "dbo:portrayer": ("{head} features a character portrayed by {tails}", "{tail} portrayed a character in {head}"),
    "dbo:album": ("{head} is included in the album {tails}", "{tail} is the album that contains {head}"),
    "dbo:artist": ("{head} features a performance by {tails}", "{tail} performed in {head}"),
}

AUX_TYPE_BY_PREFIX = (("cat:", "category"), ("res:", "resource"), ("http", "url"))


# ---------------------------------------------------------------- name helpers

def readable_name(entity_id: str) -> str:
    """'res:The_Prophet_(book)' -> 'The Prophet (book)'. URLs are returned unchanged."""
    if entity_id.startswith("http"):
        return entity_id
    for prefix in ("res:", "cat:"):
        if entity_id.startswith(prefix):
            entity_id = entity_id[len(prefix):]
            break
    return entity_id.replace("_", " ")


def aux_type_of(aux_id: str) -> str:
    for prefix, aux_type in AUX_TYPE_BY_PREFIX:
        if aux_id.startswith(prefix):
            return aux_type
    raise ValueError(f"unknown aux prefix: {aux_id!r}")


# ---------------------------------------------------------------- readers

def _data_lines(path: Path):
    """Yield lines without header and without trailing CR/LF (files may use CRLF)."""
    with open(path, encoding="utf-8", newline="") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                yield line


def read_id_map(path: Path) -> dict:
    """remap_id (int) -> original id, for '<orig> <remap>' files."""
    out = {}
    for line in _data_lines(path):
        orig, remap = line.split(" ")
        out[int(remap)] = orig
    return out


def read_items(path: Path):
    """item_list.txt: 'ori_id remap_id dbpedia_id' -> {remap: (asin, dbpedia_id)}."""
    out = {}
    for line in _data_lines(path):
        asin, remap, dbpedia = line.split(" ")
        out[int(remap)] = (asin, dbpedia)
    return out


def read_reviews(path: Path):
    """review_list.txt: 'user item review...' (review may contain spaces, may be empty)."""
    for line in _data_lines(path):
        parts = line.split(" ", 2)
        yield int(parts[0]), int(parts[1]), parts[2] if len(parts) == 3 else ""


def read_pairs(path: Path):
    for line in _data_lines(path):
        u, i = line.split(" ")
        yield int(u), int(i)


def read_kg(path: Path):
    """kg_final.txt has no header."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                h, r, t = line.split()
                yield int(h), int(r), int(t)


# ---------------------------------------------------------------- writers

def _csv_writer(f):
    return csv.writer(f, lineterminator="\n")


def convert(src: Path, out: Path) -> dict:
    users = read_id_map(src / "user_list.txt")
    items = read_items(src / "item_list.txt")
    entities = read_id_map(src / "entity_list.txt")
    relations = read_id_map(src / "relation_list.txt")
    n_items = len(items)

    # Sanity checks on the author data layout.
    assert sorted(items) == list(range(n_items)), "item remap ids are not 0..n-1"
    for idx, (_, dbpedia) in items.items():
        assert "res:" + entities[idx] == dbpedia, f"entity {idx} != item dbpedia id"
    missing = set(relations.values()) - set(RELATION_TEMPLATES)
    assert not missing, f"no template for relations: {missing}"

    pairs = list(read_pairs(src / "interaction_list.txt"))
    reviews = list(read_reviews(src / "review_list.txt"))
    assert len(pairs) == len(reviews), "interaction_list and review_list differ in length"

    out.mkdir(parents=True, exist_ok=True)
    stats = {}

    # interactions.jsonl
    n_empty = 0
    with open(out / "interactions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (u, i), (ru, ri, text) in zip(pairs, reviews):
            assert (u, i) == (ru, ri), f"review row mismatch at {(u, i)} vs {(ru, ri)}"
            if not text.strip():
                text = EMPTY_REVIEW_PLACEHOLDER
                n_empty += 1
            row = {"user_id": users[u], "item_id": items[i][0], "review": text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["interactions"] = len(pairs)
    stats["empty_reviews_filled"] = n_empty

    # items.csv
    with open(out / "items.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv_writer(f)
        w.writerow(["item_id", "name"])
        for idx in range(n_items):
            asin, dbpedia = items[idx]
            w.writerow([asin, readable_name(dbpedia)])
    stats["items"] = n_items

    # auxs.csv (entities after the items, in remap order)
    aux_ids = [entities[e] for e in sorted(entities) if e >= n_items]
    with open(out / "auxs.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv_writer(f)
        w.writerow(["aux_id", "name", "aux_type"])
        for aux_id in aux_ids:
            w.writerow([aux_id, readable_name(aux_id), aux_type_of(aux_id)])
    stats["auxs"] = len(aux_ids)

    # kg.csv (author order, duplicates kept)
    n_kg = 0
    with open(out / "kg.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv_writer(f)
        w.writerow(["head_type", "head", "relation", "tail_type", "tail"])
        for h, r, t in read_kg(src / "kg_final.txt"):
            assert h < n_items <= t, f"triple is not item -> aux: {(h, r, t)}"
            w.writerow(["item", items[h][0], relations[r], "aux", entities[t]])
            n_kg += 1
    stats["kg_rows"] = n_kg

    # relations.csv (author relation order)
    with open(out / "relations.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv_writer(f)
        w.writerow(["relation", "head_type", "tail_type", "sentence", "reverse_sentence"])
        for r in sorted(relations):
            name = relations[r]
            sentence, reverse = RELATION_TEMPLATES[name]
            w.writerow([name, "item", "aux", sentence, reverse])
    stats["relations"] = len(relations)

    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="author amazon-book_kg folder")
    p.add_argument("out", type=Path, help="output raw folder, e.g. data/book/raw")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = p.parse_args(argv)

    existing = [n for n in OUTPUT_FILES if (args.out / n).exists()]
    if existing and not args.force:
        sys.exit(f"output files already exist in {args.out}: {existing} (use --force)")

    stats = convert(args.src, args.out)
    for k, v in stats.items():
        print(f"{k:>22}: {v:,}")


if __name__ == "__main__":
    main()
