import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DAPT text data from MRI caption JSONs.")
    parser.add_argument(
        "--data_roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dataset roots containing modal_wise_finding*.json",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    total = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for root in args.data_roots:
            finding_path = root / "modal_wise_finding.json"
            finding_cap5_path = root / "modal_wise_finding_cap5.json"
            if not finding_path.exists() or not finding_cap5_path.exists():
                raise FileNotFoundError(f"Missing modal_wise_finding*.json under {root}")

            finding = load_json(finding_path)
            finding_cap5 = load_json(finding_cap5_path)

            keys = sorted(set(finding.keys()) | set(finding_cap5.keys()))
            for key in keys:
                if key in finding:
                    text = str(finding[key]).strip()
                    if text:
                        f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
                        total += 1
                if key in finding_cap5:
                    caps = finding_cap5[key]
                    if isinstance(caps, list):
                        for cap in caps:
                            text = str(cap).strip()
                            if text:
                                f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
                                total += 1
                    else:
                        text = str(caps).strip()
                        if text:
                            f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
                            total += 1

    print(f"Wrote {total} samples to {args.out}")


if __name__ == "__main__":
    main()
