# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


ROOT = Path(SPECPATH).parent

datas = (
    collect_data_files("tensorflow", include_py_files=False)
    + collect_data_files("keras", include_py_files=False)
    + collect_data_files("maze_utils", include_py_files=False)
)
binaries = (
    collect_dynamic_libs("tensorflow")
    + collect_dynamic_libs("numpy")
    + collect_dynamic_libs("scipy")
    + collect_dynamic_libs("h5py")
)
hiddenimports = [
    "tensorflow",
    "keras",
    "maze_utils",
    "scipy",
    "scipy.linalg",
    "scipy.sparse",
]

a = Analysis(
    [str(ROOT / "ui" / "tk_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "keras.src.backend.jax",
        "keras.src.backend.torch",
        "matplotlib",
        "notebook",
        "pytest",
        "tensorflow_estimator",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="reinforce-maze-lab",
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
