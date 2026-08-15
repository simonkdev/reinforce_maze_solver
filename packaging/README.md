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

If `staticx` can be installed and can wrap the executable, the final artifact is:

```text
packaging/artifacts/reinforce-maze-lab-linux-x86_64
```

That wrapped binary is the closest practical Linux target for NixOS-style portability because TensorFlow ships native shared libraries. If `staticx` is unavailable or cannot wrap the TensorFlow bundle, the script creates:

```text
packaging/artifacts/reinforce-maze-lab-linux-x86_64.tar.gz
```

Building from the local Nix/devenv shell can produce bundled libraries with Nix `DT_RUNPATH` entries. `staticx` cannot wrap those libraries reliably, so the local Nix build may fall back to the tarball. Use the GitHub Actions workflow or another minimal non-Nix Linux builder for the best chance of producing the wrapped Linux executable.

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
