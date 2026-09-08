"""Render evaluation summaries and permitted qualitative examples."""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from defense.anomaly import detect_anomalies

from .metrics import compute_metrics


def load_results(
    results_path: str | Path,
    partition: str = "validation",
    final_run: bool = False,
) -> pd.DataFrame:
    """Load only the requested partition, refusing test data before final run."""
    if partition == "test" and not final_run:
        raise PermissionError("test partition is available only with --final-run")
    results_path = Path(results_path)
    if results_path.suffix.lower() == ".jsonl":
        rows = []
        with results_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if partition == "all" or row.get("partition") in {partition, None}:
                    rows.append(row)
        return pd.DataFrame(rows)
    frame = pd.read_parquet(results_path) if results_path.suffix.lower() == ".parquet" else pd.read_csv(results_path)
    if "partition" in frame.columns and partition != "all":
        frame = frame[frame["partition"] == partition]
    return frame.reset_index(drop=True)


def render_report(
    results_path: str | Path,
    output_path: str | Path,
    image_root: str | Path | None = None,
    partition: str = "validation",
    final_run: bool = False,
) -> Path:
    """Render Markdown or HTML without consuming test rows prematurely."""
    results_path = Path(results_path)
    output_path = Path(output_path)
    frame = load_results(results_path, partition, final_run)
    metrics = compute_metrics(frame) if not frame.empty else {}
    examples = select_examples(frame)
    _ensure_heatmaps(examples, results_path.parent, image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(
                build_report_data(metrics, examples, partition, image_root),
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        return output_path
    content = render_markdown(metrics, examples, partition)
    if output_path.suffix.lower() in {".html", ".htm"}:
        output_path.write_text(_markdown_to_html(content), encoding="utf-8")
    else:
        output_path.write_text(content, encoding="utf-8")
    return output_path


def build_report_data(
    metrics: dict[str, Any],
    examples: dict[str, dict[str, Any]],
    partition: str,
    image_root: str | Path | None,
) -> dict[str, Any]:
    """Build the JSON contract consumed by the static dashboard."""
    return {
        "partition": partition,
        "image_root": str(image_root) if image_root is not None else "",
        "metrics": metrics,
        "examples": examples,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def select_examples(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select clean-pass, attack-recovered, and ambiguous-abstained examples."""
    if frame.empty:
        return {}
    clean_pass = frame[
        (frame["attack_family"] == "clean")
        & (frame["shield_decision"] == "TRUSTED")
        & (frame["shield_correct"] == True)
    ]
    recovered = frame[
        (frame["attack_family"] != "clean")
        & (frame["raw_correct"] == False)
        & (frame["shield_correct"] == True)
    ]
    abstained = frame[frame["shield_decision"] == "ABSTAIN"]
    choices = {
        "clean-pass": clean_pass,
        "attack-recovered": recovered,
        "ambiguous-abstained": abstained,
    }
    return {name: subset.iloc[0].to_dict() for name, subset in choices.items() if not subset.empty}


def render_markdown(metrics: dict[str, Any], examples: dict[str, dict[str, Any]], partition: str) -> str:
    lines = [f"# ARGUS Shield Evaluation ({partition})", "", "## Summary", ""]
    lines.extend(["| Metric | Value |", "|---|---:|"])
    if metrics:
        lines.extend(
            [
                f"| Clean accuracy, raw | {metrics['clean_accuracy']['raw']} |",
                f"| Clean accuracy, shield | {metrics['clean_accuracy']['shield']} |",
                f"| Abstention rate | {metrics['abstention_rate']['overall']} |",
                f"| Trusted accuracy at fixed FPR | {metrics['trusted_accuracy_at_fixed_false_positive_rate']['accuracy']} |",
                f"| Raw latency p50/p95 (ms) | {metrics['latency_ms']['raw']['p50']} / {metrics['latency_ms']['raw']['p95']} |",
                f"| Shield latency p50/p95 (ms) | {metrics['latency_ms']['shield']['p50']} / {metrics['latency_ms']['shield']['p95']} |",
            ]
        )
    lines.extend(["", "## Qualitative Examples", "", "| Case | Image | Raw output | Shield decision | Shield output | Heatmap overlay |", "|---|---|---|---|---|---|"])
    for case, row in examples.items():
        lines.append(
            "| {case} | {image} | {raw} | {decision} | {shield} | {heatmap} |".format(
                case=case,
                image=row.get("image_path", ""),
                raw=_detection_summary(row.get("raw_detections", [])),
                decision=row.get("shield_decision", ""),
                shield=_detection_summary(row.get("shield_detections", [])),
                heatmap=row.get("anomaly_heatmap_path", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _detection_summary(value: Any) -> str:
    if not value:
        return "none"
    return ", ".join(
        f"{item.get('label', '?')} {float(item.get('score', 0.0)):.2f}"
        for item in value
    )


def _ensure_heatmaps(examples: dict[str, dict[str, Any]], result_dir: Path, image_root: str | Path | None) -> None:
    if image_root is None:
        return
    image_root = Path(image_root)
    for row in examples.values():
        heatmap_value = row.get("anomaly_heatmap_path")
        heatmap_path = Path(heatmap_value) if heatmap_value else result_dir / "heatmaps" / f"{row['image_path']}.png"
        if not heatmap_path.is_absolute():
            heatmap_path = result_dir / heatmap_path if not heatmap_path.exists() else heatmap_path
        if heatmap_path.exists():
            row["anomaly_heatmap_path"] = str(heatmap_path)
            continue
        image = cv2.imread(str(image_root / str(row["image_path"])), cv2.IMREAD_COLOR)
        if image is None:
            continue
        anomaly = detect_anomalies(image).anomaly_map
        heatmap = cv2.normalize(np.asarray(anomaly), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(heatmap_path), cv2.addWeighted(image, 0.55, heatmap, 0.45, 0))
        row["anomaly_heatmap_path"] = str(heatmap_path)


def _markdown_to_html(markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<!doctype html><html><body><pre>{escaped}</pre></body></html>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ARGUS Shield evaluation report.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--partition", choices=["validation", "train", "test", "all"], default="validation")
    parser.add_argument("--final-run", action="store_true")
    args = parser.parse_args()
    render_report(args.results, args.output, args.image_root, args.partition, args.final_run)


if __name__ == "__main__":
    main()