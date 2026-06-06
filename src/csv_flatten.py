"""Flatten Doc2CSV-AI training CSV's `output` JSON column into separate columns.

Takes a CSV produced by Pipeline (columns: instruction, input, output, source,
chunk_id) where `output` is a JSON string, and produces a NEW CSV where each
JSON key becomes its own column. Nested objects/arrays are stringified back
to compact JSON in the cell.

Array explosion (explode_arrays=True, default):
  Any field whose parsed value is a JSON array is "exploded" — one output row
  is emitted per array element, with all other columns duplicated.
  Multiple array-valued fields in the same JSON object produce a Cartesian
  product of rows.

  Example: {"bệnh": "Sốt xuất huyết", "biện_pháp": ["diệt muỗi","ngủ màn","phun thuốc"]}
  → 3 rows, each with bệnh="Sốt xuất huyết" and biện_pháp set to one item.
"""
import csv
import itertools
import json
from pathlib import Path
from typing import Callable, Optional


def _to_cell(v) -> str:
    """Render a JSON value as a flat CSV cell."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return str(v)


def flatten_csv(
    src_path: str,
    dst_path: str,
    on_log: Optional[Callable[[str], None]] = None,
    explode_arrays: bool = True,
) -> dict:
    """Read a Doc2CSV training CSV; parse every row's `output` JSON; write a
    new CSV where each JSON key is promoted to its own column.

    Column order: instruction, input, <JSON keys in discovery order>, source, chunk_id

    When explode_arrays=True (default): any JSON field whose value is a list is
    exploded into multiple rows — one row per list element, all other columns
    duplicated. Multiple array fields in the same object produce a Cartesian
    product of rows.
    """
    log = on_log or (lambda m: None)

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")

    rows: list[dict] = []
    json_keys: list[str] = []   # preserve discovery order, union across all rows
    seen: set[str] = set()
    parse_errors = 0
    src_rows = 0          # rows in source CSV
    exploded_extra = 0    # additional rows added by array explosion

    with open(src, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_rows += 1
            raw = (row.get("output") or "").strip()
            parsed = None
            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                parsed = {"raw": raw}
                parse_errors += 1
            elif not isinstance(parsed, dict):
                parsed = {"value": parsed}

            # Collect all JSON keys in discovery order (union schema)
            for k in parsed:
                if k not in seen:
                    seen.add(k)
                    json_keys.append(k)

            base = {
                "instruction": row.get("instruction", ""),
                "input":       row.get("input", ""),
                "source":      row.get("source", ""),
                "chunk_id":    row.get("chunk_id", ""),
            }

            # Determine which fields contain arrays (candidates for explosion)
            array_keys = (
                [k for k, v in parsed.items() if isinstance(v, list)]
                if explode_arrays else []
            )

            if array_keys:
                # Scalar (non-array) fields → single cell value each
                scalar_fields: dict[str, str] = {
                    k: _to_cell(v)
                    for k, v in parsed.items()
                    if not isinstance(v, list)
                }

                # For every array field build a list of (key, cell_string) pairs
                # where cell_string is the string form of a SINGLE element.
                per_key: list[list[tuple[str, str]]] = []
                for k in array_keys:
                    lst = parsed[k]
                    if lst:
                        per_key.append([(k, _to_cell(el)) for el in lst])
                    else:
                        per_key.append([(k, "")])   # empty array → one empty row

                # Cartesian product across all array fields
                combos = list(itertools.product(*per_key))
                for combo in combos:
                    fields = dict(scalar_fields)
                    for k, cell_val in combo:
                        fields[k] = cell_val
                    rows.append({**base, "_fields": fields})

                exploded_extra += len(combos) - 1   # -1: first combo replaces original row

            else:
                # No arrays (or explosion disabled) — single output row
                fields = {k: _to_cell(v) for k, v in parsed.items()}
                rows.append({**base, "_fields": fields})

    columns = ["instruction", "input"] + json_keys + ["source", "chunk_id"]

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            out_row = {
                "instruction": r["instruction"],
                "input":       r["input"],
                "source":      r["source"],
                "chunk_id":    r["chunk_id"],
            }
            for k in json_keys:
                out_row[k] = r["_fields"].get(k, "")
            writer.writerow(out_row)

    explode_note = f" · +{exploded_extra} dòng array explosion" if exploded_extra else ""
    log(
        f"   ✓ {src.name}: {src_rows} nguồn → {len(rows)} dòng"
        f" · {len(json_keys)} cột JSON · {parse_errors} lỗi parse"
        f"{explode_note} → {dst.name}"
    )
    return {
        "rows":          len(rows),
        "src_rows":      src_rows,
        "json_columns":  len(json_keys),
        "json_keys":     json_keys,
        "parse_errors":  parse_errors,
        "exploded_rows": exploded_extra,
        "dst":           str(dst),
    }
