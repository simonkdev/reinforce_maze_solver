$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$BuildDir = Join-Path $Root "packaging\build\pyinstaller"
$DistDir = Join-Path $Root "packaging\dist"
$ArtifactDir = Join-Path $Root "packaging\artifacts"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r packaging/requirements-packaging.txt

python -m PyInstaller --clean --noconfirm --workpath $BuildDir --distpath $DistDir packaging/reinforce_maze_lab.spec

New-Item -ItemType Directory -Force $ArtifactDir | Out-Null
Copy-Item -Force (Join-Path $DistDir "reinforce-maze-lab.exe") (Join-Path $ArtifactDir "reinforce-maze-lab-windows-x86_64.exe")
Get-Item (Join-Path $ArtifactDir "reinforce-maze-lab-windows-x86_64.exe")
