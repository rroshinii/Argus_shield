.PHONY: install test run-api run-eval

MANIFEST ?= data/manifests/eval.jsonl
IMAGE_ROOT ?= data/clean
OUTPUT_DIR ?= eval/results

install:
	python -m pip install -e .

test:
	python -m pytest

run-api:
	python -m uvicorn api.main:app --reload

run-eval:
	python -m eval.run_eval --manifest $(MANIFEST) --image-root $(IMAGE_ROOT) --output-dir $(OUTPUT_DIR)
