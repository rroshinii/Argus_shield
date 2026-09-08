# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

datas = [
    ('../configs', 'configs'),
    ('../docs', 'docs'),
    ('../ui/static', 'ui/static'),
    ('../data', 'data'),
]

hiddenimports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.lifespans',
    'uvicorn.lifespans.on',
    'fastapi',
    'starlette',
    'starlette.staticfiles',
    'starlette.responses',
    'starlette.middleware',
    'starlette.middleware.base',
    'pydantic',
    'pydantic_core',
    'torch',
    'torchvision',
    'torchvision.models.detection',
    'torchvision.models.detection.faster_rcnn',
    'PIL',
    'cv2',
    'yaml',
    'webview',
    'clr_loader',
    'pythonnet',
    'scipy',
    'sklearn',
    'pandas',
    'numpy',
]

a = Analysis(
    ['entrypoint.py'],
    pathex=['..'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ARGUSShield',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
