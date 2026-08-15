# Packaging

This project packages the Tkinter UI entry point, `ui/tk_app.py`, without changing the RL source code.

## Linux

Run from the repository root:

```bash
scripts/build_linux.sh
```

The script creates a clean virtual environment under `packaging/build/venv` before installing dependencies. That keeps the package graph independent from the active shell environment.

The script builds a PyInstaller one-file executable at:

```text
packaging/dist/reinforce-maze-lab
```

The published Linux artifact is copied to:

```text
packaging/artifacts/reinforce-maze-lab-linux-x86_64
```

The Linux artifact is intended for normal Linux distributions. It may be dynamically linked. The GitHub Actions Linux job runs on Ubuntu 22.04 in a fresh Python environment and fails only if `ldd` reports missing shared libraries.

## Windows

Build on Windows, not by cross-compiling from Linux:

```powershell
scripts\build_windows.ps1
```

The output is a portable executable, not an installer:

```text
packaging\artifacts\reinforce-maze-lab-windows-x86_64.exe
```

The Windows executable intentionally opens a command-line window. TensorFlow startup can take a while, and the console makes import progress and crashes visible instead of looking like a silent hang.

TensorFlow publishes Windows CPU wheels for modern Python versions, so the Windows build installs the normal project dependencies with pip and bundles them into the executable.
