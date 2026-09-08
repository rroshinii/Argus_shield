# ARGUS Shield — Packaging & Deployment Guide

This directory contains configuration, specifications, and build scripts for packaging ARGUS Shield into a standalone, single-file Windows desktop application (`ARGUSShield.exe`).

---

## 1. Overview

The packaged application combines:
- **FastAPI Backend Server**: Running locally via embedded Uvicorn.
- **Defense Pipeline**: Multi-view anomaly detection, image inpainting/reconstruction, and consistency cross-checking.
- **Desktop UI Shell**: Native Windows window rendered with `pywebview` (zero Node.js/Electron dependencies).
- **Embedded UI Static Assets**: Dark technical instrument panel (`ui/static`).

On launch, `ARGUSShield.exe` automatically binds to a free local port (default: 8765), starts the background inference engine, and presents the desktop GUI without requiring a console window, external Python runtime, or any `pip install` steps.

---

## 2. Adapter Modes: MOCK vs. LIVE

### Default Behavior: MOCK Mode
By default, the packaged executable ships in **MOCK MODE** using `torchvision.models.detection.fasterrcnn_resnet50_fpn`. A visible `MOCK MODE` warning badge is prominently displayed across the header and telemetry panels to ensure simulated test predictions are never mistaken for live model output.

### Switching to Live ARGUS Endpoint
To connect ARGUS Shield to a live black-box ARGUS vision API endpoint:

1. Edit [`configs/task_contract.yaml`](../configs/task_contract.yaml):
   ```yaml
   adapter: http
   base_url: "https://your-argus-api-endpoint.example.com/v1"
   access_level: "black-box"
   ```
2. Set the environment variable `ARGUS_API_KEY`:
   ```powershell
   $env:ARGUS_API_KEY = "your-api-key-here"
   ```
3. Rebuild the standalone executable:
   ```cmd
   packaging\build.bat
   ```
   Or launch dynamically via Python:
   ```powershell
   python -m ui.desktop
   ```

---

## 3. Building the Executable

### Prerequisites
Ensure dependencies are installed in your Python 3.11 environment:
```powershell
pip install -e .
pip install pyinstaller pywebview
```

### Build Command
Run the build script:
```cmd
packaging\build.bat
```

The output executable will be created at:
```
packaging\dist\ARGUSShield.exe
```

---

## 4. Verification

After building, double-click `ARGUSShield.exe`:
1. The desktop window titled **"ARGUS Shield — Defense Boundary"** appears.
2. The persistent badge in the top bar indicates `MOCK MODE: FASTER-RCNN`.
3. Drop an image into the upload drop-zone (or click a sample button) to verify real-time inference and bounding box visualization.
4. Toggle the **Show Anomaly Heatmap** button to view spatial anomaly overlays.
