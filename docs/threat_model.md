# ARGUS Shield Threat Model

## Purpose

ARGUS Shield protects an image inference task from manipulated or unstable inputs. The MVP is an evaluation and serving boundary, not a claim that every semantic or distribution-shift failure can be detected.

## Manipulation Classes

| Class | Example | MVP status |
|---|---|---|
| Pixel perturbation | Bounded L-infinity or L2 black-box random noise | **Measured by the harness when included in a manifest.** No gradient or logit access is assumed. |
| Patch | Solid/checkerboard patch or a pasted naturalistic crop | **Measured by the harness when included in a manifest.** Cheap local anomaly features and recovery are intended to surface localized regions. |
| Geometric/photometric | Resize, crop, rotation, blur, JPEG recompression, brightness/contrast shifts, print-recapture stress | **Measured by the harness when included in a manifest.** These are treated as benign-but-stressing transformations. |
| Semantic | Object replacement, compositing with task-level meaning changes, or content that remains locally natural | **Future work.** The MVP has no semantic detector, task-specific grounder, or confirmed ARGUS contract for this class. |

## Defense Boundary

The production path is `api/` -> `defense/pipeline.py` -> `adapter/`. The `attacks/` package is excluded from that path and exists only to create controlled evaluation inputs. The pipeline first runs cheap anomaly heuristics, then conditionally runs multi-view inference, recovery, and consistency checks. It can return `ABSTAIN` rather than force a low-confidence label.

## Signals and Responses

- DCT high-frequency energy, blurred reconstruction error, and local texture variance provide inexpensive local evidence.
- Suspicious regions can be blurred or inpainted before a second inference pass.
- Agreement across original, transformed, denoised, recompressed, cropped, and recovered views is used as a stability signal.
- The decision policy returns `TRUSTED`, `FLAGGED_WITH_WARNING`, or `ABSTAIN` using values loaded from `configs/defense_thresholds.yaml`.

## Out of Scope for This MVP

The MVP does not guarantee robustness to semantic manipulation, adaptive attacks that optimize against the complete deployed pipeline, model extraction, poisoned training data, or a mismatch between the assumed task contract and the real ARGUS service. Those cases require a confirmed task definition, broader datasets, and task-specific defenses.
