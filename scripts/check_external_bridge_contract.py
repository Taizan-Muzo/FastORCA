"""
Lightweight External Bridge Contract Smoke Checker.

Checks unified JSON files for external bridge contract integrity without running
any heavy computation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

EXTERNAL_BRIDGE_EXECUTION_STATUS_ENUM = {
    "not_attempted",
    "success",
    "failed",
    "timeout",
    "skipped",
    "disabled",
}

ROADMAP_STATUS_ENUM = {
    "missing",
    "placeholder",
    "implemented_proxy",
    "implemented_exact",
}

DEPRECATED_BRIDGE_FIELDS = {
    "input_file",
    "output_file",
    "execution_time_seconds",
    "critic2_version",
}

REQUIRED_BRIDGE_FIELDS = {
    "execution_status",
    "failure_reason",
    "metadata",
    "artifact_refs",
    "warnings",
}

ROADMAP_ALLOWED_FIELDS = {
    "status",
    "payload",
    "needs_exact_qcmol_name",
    "notes",
}

FORBIDDEN_EXECUTION_FIELDS = {
    "execution_status",
    "failure_reason",
    "artifact_refs",
    "command",
    "environment",
    "stdout_file",
    "stderr_file",
    "input_file",
    "output_file",
    "execution_time_seconds",
    "tool_version",
}

SCIENTIFIC_KEY_HINTS = {
    "qtaim",
    "charges",
    "charge",
    "matrix",
    "descriptor",
    "descriptors",
    "bader",
    "payload",
    "features",
}


def issue(file: str, rule_id: str, json_path: str, message: str, severity: str) -> Dict[str, str]:
    return {
        "file": file,
        "rule_id": rule_id,
        "json_path": json_path,
        "message": message,
        "severity": severity,  # "error" | "warning"
    }


def _walk_dict(node: Any, prefix: str) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append((p, v))
            out.extend(_walk_dict(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            p = f"{prefix}[{i}]"
            out.extend(_walk_dict(v, p))
    return out


def validate_unified_contract(data: Dict[str, Any], file_path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    violations: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    external_bridge = data.get("external_bridge")
    external_features = data.get("external_features")
    roadmap = data.get("external_bridge_roadmap")

    if not isinstance(external_bridge, dict):
        violations.append(issue(file_path, "RULE_2", "external_bridge", "external_bridge must be an object", "error"))
        external_bridge = {}
    if not isinstance(external_features, dict):
        violations.append(issue(file_path, "RULE_3", "external_features", "external_features must be an object", "error"))
        external_features = {}
    if not isinstance(roadmap, dict):
        violations.append(issue(file_path, "RULE_5", "external_bridge_roadmap", "external_bridge_roadmap must be an object", "error"))
        roadmap = {}

    # Rule 1 + Rule 2 + Rule 4 + Rule 7
    for tool, node in external_bridge.items():
        path = f"external_bridge.{tool}"
        if not isinstance(node, dict):
            violations.append(issue(file_path, "RULE_2", path, "tool node must be an object", "error"))
            continue

        status = node.get("execution_status")
        if status not in EXTERNAL_BRIDGE_EXECUTION_STATUS_ENUM:
            violations.append(
                issue(
                    file_path,
                    "RULE_1",
                    f"{path}.execution_status",
                    f"invalid execution_status: {status}",
                    "error",
                )
            )

        missing_new = sorted(k for k in REQUIRED_BRIDGE_FIELDS if k not in node)
        if missing_new:
            violations.append(
                issue(
                    file_path,
                    "RULE_2",
                    path,
                    f"missing required contract fields: {missing_new}",
                    "error",
                )
            )

        if "metadata" in node and not isinstance(node.get("metadata"), dict):
            violations.append(issue(file_path, "RULE_2", f"{path}.metadata", "metadata must be an object", "error"))
        if "artifact_refs" in node and not isinstance(node.get("artifact_refs"), dict):
            violations.append(issue(file_path, "RULE_2", f"{path}.artifact_refs", "artifact_refs must be an object", "error"))
        if "warnings" in node and not isinstance(node.get("warnings"), list):
            violations.append(issue(file_path, "RULE_2", f"{path}.warnings", "warnings must be a list", "error"))

        has_deprecated = any(k in node for k in DEPRECATED_BRIDGE_FIELDS)
        has_new_all = all(k in node for k in REQUIRED_BRIDGE_FIELDS)
        if has_deprecated and not has_new_all:
            violations.append(
                issue(
                    file_path,
                    "RULE_7",
                    path,
                    "deprecated fields cannot be the only contract source; required new fields are missing",
                    "error",
                )
            )
        elif has_deprecated and has_new_all:
            warnings.append(
                issue(
                    file_path,
                    "RULE_7",
                    path,
                    "deprecated fields present (compatibility mode); prefer new metadata/artifact_refs fields",
                    "warning",
                )
            )

        # Rule 4: no scientific payload in external_bridge tool node
        allowed = REQUIRED_BRIDGE_FIELDS | DEPRECATED_BRIDGE_FIELDS
        for k in node.keys():
            if k in allowed:
                continue
            lk = k.lower()
            if any(h in lk for h in SCIENTIFIC_KEY_HINTS):
                violations.append(
                    issue(
                        file_path,
                        "RULE_4",
                        f"{path}.{k}",
                        "scientific payload must not be stored in external_bridge; move to external_features",
                        "error",
                    )
                )

    # Rule 3: execution-layer keys must not leak into external_features
    for p, _ in _walk_dict(external_features, "external_features"):
        key = p.split(".")[-1]
        key = key.split("[")[0]
        if key in FORBIDDEN_EXECUTION_FIELDS:
            violations.append(
                issue(
                    file_path,
                    "RULE_3",
                    p,
                    f"{key} is not allowed in external_features",
                    "error",
                )
            )

    # Rule 5 + Rule 6
    for level, level_node in roadmap.items():
        level_path = f"external_bridge_roadmap.{level}"
        if not isinstance(level_node, dict):
            violations.append(issue(file_path, "RULE_5", level_path, "roadmap level must be an object", "error"))
            continue
        for feat, placeholder in level_node.items():
            ph_path = f"{level_path}.{feat}"
            if not isinstance(placeholder, dict):
                violations.append(issue(file_path, "RULE_5", ph_path, "placeholder must be an object", "error"))
                continue

            keys = set(placeholder.keys())
            extras = sorted(keys - ROADMAP_ALLOWED_FIELDS)
            missing = sorted(ROADMAP_ALLOWED_FIELDS - keys)
            if extras:
                violations.append(
                    issue(
                        file_path,
                        "RULE_5",
                        ph_path,
                        f"placeholder has forbidden fields: {extras}",
                        "error",
                    )
                )
            if missing:
                violations.append(
                    issue(
                        file_path,
                        "RULE_5",
                        ph_path,
                        f"placeholder missing required fields: {missing}",
                        "error",
                    )
                )

            status = placeholder.get("status")
            if status not in ROADMAP_STATUS_ENUM:
                violations.append(
                    issue(
                        file_path,
                        "RULE_6",
                        f"{ph_path}.status",
                        f"invalid roadmap status: {status}",
                        "error",
                    )
                )

            # Type checks (payload/notes may be null)
            if "needs_exact_qcmol_name" in placeholder and not isinstance(placeholder["needs_exact_qcmol_name"], bool):
                violations.append(
                    issue(
                        file_path,
                        "RULE_5",
                        f"{ph_path}.needs_exact_qcmol_name",
                        "needs_exact_qcmol_name must be boolean",
                        "error",
                    )
                )
            if "notes" in placeholder and placeholder["notes"] is not None and not isinstance(placeholder["notes"], str):
                violations.append(issue(file_path, "RULE_5", f"{ph_path}.notes", "notes must be string or null", "error"))

    return violations, warnings


def collect_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files = sorted(path.rglob("*.unified.json"))
    if files:
        return files
    return sorted(path.rglob("*.json"))


def run_check(target: Path) -> Dict[str, Any]:
    files = collect_files(target)
    result: Dict[str, Any] = {
        "checked_files": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0,
        "violations": [],
    }

    for fp in files:
        result["checked_files"] += 1
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            result["failed"] += 1
            result["violations"].append(
                issue(str(fp), "RULE_PARSE", "$", f"failed to load JSON: {e}", "error")
            )
            continue

        violations, warnings = validate_unified_contract(data, str(fp))
        result["violations"].extend(violations)
        result["violations"].extend(warnings)
        result["warnings"] += len(warnings)
        if violations:
            result["failed"] += 1
        else:
            result["passed"] += 1

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check external bridge contract for unified JSON files")
    p.add_argument("target", help="Path to a unified JSON file or a directory")
    p.add_argument("--json-out", default="", help="Optional output path for JSON summary")
    p.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as non-zero exit condition",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.target).resolve()
    result = run_check(target)

    print(
        f"Checked {result['checked_files']} file(s): "
        f"{result['passed']} passed, {result['failed']} failed, {result['warnings']} warning(s)"
    )
    for it in result["violations"][:50]:
        sev = it.get("severity", "error").upper()
        print(f"[{sev}] {it['rule_id']} {it['json_path']} - {it['message']} ({it['file']})")

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report saved: {out_path}")

    has_errors = any(v.get("severity") == "error" for v in result["violations"])
    has_warnings = any(v.get("severity") == "warning" for v in result["violations"])
    if has_errors:
        return 1
    if args.strict_warnings and has_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
