"""
Emit qcMol alignment master table snapshot (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (  # noqa: E402
    EXACT_ONLY_ARCHIVED_LIST,
    QCMOL_ALIGNMENT_MASTER_TABLE,
    remaining_alignment_gaps,
    summarize_alignment_next_actions,
    summarize_alignment_status_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export qcMol alignment master table")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    status_counts = summarize_alignment_status_counts(
        [{"status": row.get("current_status")} for row in QCMOL_ALIGNMENT_MASTER_TABLE]
    )
    action_counts = summarize_alignment_next_actions(QCMOL_ALIGNMENT_MASTER_TABLE)
    gaps = remaining_alignment_gaps(QCMOL_ALIGNMENT_MASTER_TABLE)

    payload = {
        "alignment_master_table": QCMOL_ALIGNMENT_MASTER_TABLE,
        "status_counts": status_counts,
        "next_action_counts": action_counts,
        "remaining_gap_count": len(gaps),
        "remaining_gaps": gaps,
        "exact_only_archived_list": EXACT_ONLY_ARCHIVED_LIST,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# qcMol Alignment Master Table Snapshot",
        "",
        "## Status Counts",
        (
            f"- implemented_exact={status_counts['implemented_exact']}, "
            f"implemented_proxy={status_counts['implemented_proxy']}, "
            f"partial={status_counts['partial']}, "
            f"missing={status_counts['missing']}, "
            f"rejected_as_exact={status_counts['rejected_as_exact']}"
        ),
        "",
        "## Next Action Counts",
        (
            f"- keep={action_counts['keep']}, "
            f"upgrade={action_counts['upgrade']}, "
            f"redefine={action_counts['redefine']}, "
            f"roadmap_only={action_counts['roadmap_only']}, "
            f"reject={action_counts['reject']}"
        ),
        "",
        f"## Remaining Gaps ({len(gaps)})",
    ]
    for row in gaps:
        lines.append(
            f"- {row['section']}.{row['qcMol_item_name']}: status={row['current_status']}, "
            f"next_action={row['next_action']}, mapped_path={row['mapped_path']}"
        )
    lines.extend(["", f"## Exact-Only Archived ({len(EXACT_ONLY_ARCHIVED_LIST)})"])
    for row in EXACT_ONLY_ARCHIVED_LIST:
        lines.append(
            f"- {row['qcMol_item_name']}: status={row['status']}, next_action={row['next_action']}, mapped_path={row['mapped_path']}"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(out_json),
                "output_md": str(out_md),
                "remaining_gap_count": len(gaps),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
