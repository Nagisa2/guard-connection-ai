from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from guard_connection_ai.data.yamaguchi_json import audit_yamaguchi_export

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="山口大学JSON exportの前段AI実行可否を非破壊で監査する。"
    )
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data/json_Data"
    )
    args = parser.parse_args()
    print(json.dumps(asdict(audit_yamaguchi_export(args.data_root)), indent=2))


if __name__ == "__main__":
    main()
