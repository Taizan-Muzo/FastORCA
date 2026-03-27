"""
Export consumer-friendly qcMol substitute views from unified outputs.

Produces three lightweight datasets:
- minimal
- enhanced
- caution
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (  # noqa: E402
    CONSUMER_CAUTION_FIELDS,
    CONSUMER_ENHANCED_FIELDS,
    CONSUMER_MINIMAL_FIELDS,
)


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _iter_unified_files(unified_dir: Path) -> Iterable[Path]:
    return sorted(unified_dir.glob("*.unified.json"))


def _build_row(data: Dict[str, Any], field_paths: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for path in field_paths:
        row[path] = _get(data, path)
    return row


def _dump_jsonl(rows: List[Dict[str, Any]], out_fp: Path) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    out_fp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _dump_csv(rows: List[Dict[str, Any]], out_fp: Path) -> None:
    if not rows:
        out_fp.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with out_fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    cooked[k] = json.dumps(v, ensure_ascii=False)
                else:
                    cooked[k] = v
            writer.writerow(cooked)


def _build_view_rows(unified_files: Iterable[Path], fields: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in unified_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        row = {
            "molecule_id": _get(data, "molecule_info.molecule_id") or fp.stem.replace(".unified", ""),
            "source_file": str(fp),
        }
        row.update(_build_row(data, fields))
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export minimal/enhanced/caution consumer views")
    p.add_argument("--unified-dir", required=True, help="Directory containing *.unified.json")
    p.add_argument("--output-dir", required=True, help="Output directory for exported views")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    unified_dir = Path(args.unified_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not unified_dir.exists():
        raise FileNotFoundError(f"unified-dir not found: {unified_dir}")
    files = list(_iter_unified_files(unified_dir))
    if not files:
        raise ValueError(f"No unified files found in {unified_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    views = {
        "minimal": CONSUMER_MINIMAL_FIELDS,
        "enhanced": CONSUMER_ENHANCED_FIELDS,
        "caution": CONSUMER_CAUTION_FIELDS,
    }

    result: Dict[str, Any] = {
        "input_unified_dir": str(unified_dir),
        "n_files": len(files),
        "exports": {},
    }
    for name, fields in views.items():
        rows = _build_view_rows(files, fields)
        out_jsonl = output_dir / f"qcmol_consumer_{name}.jsonl"
        out_csv = output_dir / f"qcmol_consumer_{name}.csv"
        _dump_jsonl(rows, out_jsonl)
        _dump_csv(rows, out_csv)
        result["exports"][name] = {
            "field_count": len(fields),
            "jsonl": str(out_jsonl),
            "csv": str(out_csv),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

