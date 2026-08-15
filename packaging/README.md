# Packaging

This project packages the Tkinter UI entry point, `ui/tk_app.py`, without changing the RL source code.

## Linux

Run from the repository root:

```bash
scripts/build_linux.sh
```

The script builds a PyInstaller one-file executable at:

```text
packaging/dist/reinforce-maze-lab
```

The published Linux artifact must be a `staticx` wrapped executable:

```text
packaging/artifacts/reinforce-maze-lab-linux-x86_64
```

The build fails if `staticx` cannot wrap the executable or if `ldd` does not report `not a dynamic executable`. Building from the local Nix/devenv shell can produce bundled libraries with Nix `DT_RUNPATH` entries that `staticx` refuses to wrap. Use the GitHub Actions workflow or another minimal non-Nix Linux builder for the Linux release artifact.

## Windows

Build on Windows, not by cross-compiling from Linux:

```powershell
scripts\build_windows.ps1
```

The output is a portable executable, not an installer:

```text
packaging\artifacts\reinforce-maze-lab-windows-x86_64.exe
```

TensorFlow publishes Windows CPU wheels for modern Python versions, so the Windows build installs the normal project dependencies with pip and bundles them into the executable.
