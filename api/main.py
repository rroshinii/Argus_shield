"""FastAPI boundary for ARGUS Shield inference."""

import asyncio
import base64
import binascii
import os
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from adapter.factory import get_adapter
from defense.decision import load_thresholds
from defense.pipeline import ShieldPipeline

from .middleware import ShieldMiddleware
from .schemas import Base64ImageRequest, ShieldResponse

os.environ.setdefault("SHIELD_API_KEY", "argus-shield-local")
MAX_REQUEST_BYTES = 10 * 1024 * 1024
CONTRACT_PATH = Path(__file__).resolve().parent.parent / "docs" / "task_contract.md"
THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "configs" / "defense_thresholds.yaml"


def _latency_budget_seconds() -> float:
	text = CONTRACT_PATH.read_text(encoding="utf-8")
	match = re.search(r"latency_budget_ms`?\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
	return float(match.group(1)) / 1000 if match else 0.25


def _execution_timeout_seconds() -> float:
	if os.getenv("SHIELD_ENFORCE_TIMEOUT") == "1":
		return _latency_budget_seconds()
	return float(os.getenv("SHIELD_TIMEOUT_SECONDS", "30.0"))


@lru_cache(maxsize=1)
def get_pipeline() -> ShieldPipeline:
	config_path = os.getenv("ARGUS_ADAPTER_CONFIG")
	adapter = get_adapter(config_path) if config_path else get_adapter()
	return ShieldPipeline(adapter, thresholds=str(THRESHOLDS_PATH))


@lru_cache(maxsize=1)
def get_adapter_type() -> str:
	return type(get_pipeline().adapter).__name__.replace("ArgusAdapter", "").lower()


def _package_version() -> str:
	try:
		return version("argus-shield")
	except PackageNotFoundError:
		return "0.1.0"


app = FastAPI(title="argus-shield", version=_package_version())
app.add_middleware(ShieldMiddleware)


@app.post("/shield/detect", response_model=ShieldResponse)
@app.post("/shield/predict", response_model=ShieldResponse, include_in_schema=False)
async def predict(
	request: Request,
	response: Response,
	image: UploadFile | None = File(default=None),
	payload: Base64ImageRequest | None = Body(default=None),
) -> ShieldResponse | JSONResponse:
	"""Decode an upload or base64 body, then run the configured shield."""
	content_length = request.headers.get("content-length")
	if content_length and int(content_length) > MAX_REQUEST_BYTES:
		raise HTTPException(status_code=413, detail="request exceeds maximum size")
	if image is not None:
		data = await image.read(MAX_REQUEST_BYTES + 1)
		if len(data) > MAX_REQUEST_BYTES:
			raise HTTPException(status_code=413, detail="image exceeds maximum size")
	elif payload is not None:
		data = _decode_base64(payload.image_base64)
		if len(data) > MAX_REQUEST_BYTES:
			raise HTTPException(status_code=413, detail="image exceeds maximum size")
	else:
		raise HTTPException(status_code=400, detail="provide an uploaded image or image_base64 JSON body")
	decoded, sx, sy = _decode_image(data)
	try:
		result = await asyncio.wait_for(
			asyncio.to_thread(get_pipeline().run, decoded),
			timeout=_execution_timeout_seconds(),
		)
	except asyncio.TimeoutError:
		return JSONResponse(status_code=504, content={"detail": "shield request timed out"})
	if sx != 1.0 or sy != 1.0:
		for det in result.detections:
			x1, y1, x2, y2 = det.box
			det.box = [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]
	result_response = ShieldResponse.from_result(result, result.anomaly_map)
	response.headers["X-Shield-Decision"] = (
		result_response.detections[0].decision if result_response.detections else "ABSTAIN"
	)
	return result_response


@app.get("/shield/health")
async def health() -> dict[str, object]:
	return {
		"status": "ok",
		"adapter_type": get_adapter_type(),
		"thresholds": load_thresholds(str(THRESHOLDS_PATH)),
		"version": _package_version(),
	}


@app.get("/shield/config")
async def config() -> dict[str, object]:
	with THRESHOLDS_PATH.open("r", encoding="utf-8") as stream:
		return {"thresholds": yaml.safe_load(stream) or {}}


@app.get("/shield/eval-summary")
async def eval_summary() -> dict[str, object]:
	results_dir = Path(__file__).resolve().parent.parent / "eval" / "results"
	results_path = results_dir / "results.jsonl"
	if not results_path.exists():
		raise HTTPException(status_code=404, detail="no evaluation results found")
	import json
	rows = []
	with results_path.open("r", encoding="utf-8") as stream:
		for line in stream:
			if line.strip():
				rows.append(json.loads(line))
	if not rows:
		raise HTTPException(status_code=404, detail="empty evaluation results")

	clean_rows = [r for r in rows if r.get("attack_family") == "clean"]
	clean_raw_acc = float(np.mean([r.get("raw_correct", False) for r in clean_rows])) if clean_rows else None
	clean_shield_acc = float(np.mean([r.get("shield_correct", False) for r in clean_rows])) if clean_rows else None
	clean_fpr = float(np.mean([r.get("shield_decision") == "FLAGGED_WITH_WARNING" for r in clean_rows])) if clean_rows else None
	overall_abstain = float(np.mean([r.get("shield_decision") == "ABSTAIN" for r in rows])) if rows else None

	table_rows = [
		{
			"family_or_metric": "Clean Accuracy",
			"raw_val": f"{(clean_raw_acc * 100):.1f}%" if clean_raw_acc is not None else "N/A",
			"shield_val": f"{(clean_shield_acc * 100):.1f}%" if clean_shield_acc is not None else "N/A",
			"context": "Clean baseline (target FPR <= 5%)",
		}
	]
	families = sorted({r.get("attack_family") for r in rows if r.get("attack_family") != "clean"})
	for fam in families:
		fam_rows = [r for r in rows if r.get("attack_family") == fam]
		raw_acc = float(np.mean([r.get("raw_correct", False) for r in fam_rows])) if fam_rows else 0.0
		shield_acc = float(np.mean([r.get("shield_correct", False) for r in fam_rows])) if fam_rows else 0.0
		abstain_rate = float(np.mean([r.get("shield_decision") == "ABSTAIN" for r in fam_rows])) if fam_rows else 0.0
		table_rows.append({
			"family_or_metric": f"Attack: {fam}",
			"raw_val": f"{(raw_acc * 100):.1f}%",
			"shield_val": f"{(shield_acc * 100):.1f}%",
			"context": f"Abstention rate: {(abstain_rate * 100):.1f}%",
		})
	table_rows.append({
		"family_or_metric": "Abstention Rate (Overall)",
		"raw_val": "0.0%",
		"shield_val": f"{(overall_abstain * 100):.1f}%" if overall_abstain is not None else "0.0%",
		"context": "Fail-safe abstention on suspicious inputs",
	})
	table_rows.append({
		"family_or_metric": "Clean False-Positive Rate",
		"raw_val": "0.0%",
		"shield_val": f"{(clean_fpr * 100):.1f}%" if clean_fpr is not None else "0.0%",
		"context": "Budget limit <= 5.0%",
	})
	shield_latencies = [float(r.get("latency_ms", 0.0)) for r in rows if "latency_ms" in r]
	raw_latencies = [float(r.get("raw_latency_ms", 0.0)) for r in rows if "raw_latency_ms" in r]
	if shield_latencies:
		table_rows.append({
			"family_or_metric": "Latency p50 / p95",
			"raw_val": f"{np.percentile(raw_latencies, 50):.1f} / {np.percentile(raw_latencies, 95):.1f} ms" if raw_latencies else "-",
			"shield_val": f"{np.percentile(shield_latencies, 50):.1f} / {np.percentile(shield_latencies, 95):.1f} ms",
			"context": "Budget: 250 ms",
		})
	return {
		"rows": table_rows,
		"latency_series": shield_latencies[:50],
	}


@app.post("/shield/run-eval")
async def run_evaluation() -> dict[str, str]:
	import subprocess
	import sys
	base_dir = Path(__file__).resolve().parent.parent
	cmd = [
		sys.executable,
		"-m", "eval.run_eval",
		"--manifest", str(base_dir / "data" / "manifests" / "eval.jsonl"),
		"--image-root", str(base_dir / "data" / "clean"),
		"--output-dir", str(base_dir / "eval" / "results"),
	]
	proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
	if proc.returncode != 0:
		raise HTTPException(status_code=500, detail=f"evaluation failed: {proc.stderr}")
	return {"status": "success", "detail": "evaluation finished"}


UI_STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
if UI_STATIC_DIR.exists():
	from fastapi.responses import FileResponse
	from fastapi.staticfiles import StaticFiles

	app.mount("/static", StaticFiles(directory=str(UI_STATIC_DIR)), name="static")

	@app.get("/")
	@app.get("/ui")
	async def serve_ui() -> FileResponse:
		return FileResponse(UI_STATIC_DIR / "index.html")


def _decode_base64(value: str) -> bytes:
	if "," in value and value.startswith("data:"):
		value = value.split(",", 1)[1]
	try:
		return base64.b64decode(value, validate=True)
	except (binascii.Error, ValueError) as exc:
		raise HTTPException(status_code=400, detail="invalid base64 image") from exc


def _decode_image(data: bytes, max_dimension: int = 800) -> tuple[np.ndarray, float, float]:
	image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
	if image is None:
		raise HTTPException(status_code=400, detail="uploaded data is not a readable image")
	orig_h, orig_w = image.shape[:2]
	if max(orig_h, orig_w) > max_dimension:
		scale = max_dimension / float(max(orig_h, orig_w))
		new_w = max(1, int(round(orig_w * scale)))
		new_h = max(1, int(round(orig_h * scale)))
		image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		sx = orig_w / float(new_w)
		sy = orig_h / float(new_h)
		return image, sx, sy
	return image, 1.0, 1.0


