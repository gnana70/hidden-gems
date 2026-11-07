# Hidden Gems / Idol: runs the whole test on this PC's NVIDIA GPU inside WSL2 (Ubuntu).
#   .\Run-Idol-Tests.ps1      (from the 01-idol folder)
# First run: installs the environment, downloads ~6.8 GB of weights, then runs ~20 generations (30-45 min on our laptop).
# Close games / GPU-heavy apps first: Idol peaks near 7.2 GB (5.4 GB on our 8 GB laptop GPU).
$ErrorActionPreference = "Stop"
$lab = (wsl -e wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim()
wsl -e bash -lc "cd '$lab' && bash run_all.sh"
