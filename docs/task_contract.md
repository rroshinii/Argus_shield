# Task Contract

This document defines the task-level contract for an ARGUS Shield evaluation.

- `task_type`: object detection (ASSUMPTION; confirm with ARGUS owner)
- `input_format`: image/png-or-jpeg (ASSUMPTION; confirm with ARGUS owner)
- `output_schema`: detections with label, score, and `[x1,y1,x2,y2]` box (ASSUMPTION)
- `access_level`: mock locally, black-box HTTP remotely (ASSUMPTION)
- `query_budget`: 1000 (ASSUMPTION)
- `latency_budget_ms`: 250
- `false_positive_rate_budget`: 0.05
- `abstention_allowed`: true (ASSUMPTION)
- `submission_format`: JSON (ASSUMPTION)
