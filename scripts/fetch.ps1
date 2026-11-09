# Usage: .\scripts\fetch.ps1 lexsi-day1-checks day1
param([string]$Kernel, [string]$Dest)
$R = Split-Path $PSScriptRoot -Parent
kaggle kernels status "shivendushivu/$Kernel"
kaggle kernels output "shivendushivu/$Kernel" -p "$R\results\$Dest" --force
