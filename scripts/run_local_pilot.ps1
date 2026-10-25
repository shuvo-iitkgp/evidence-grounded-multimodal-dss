param(
    [ValidateSet('mediaeval','fakeddit','weibo')][string]$Dataset = 'fakeddit',
    [string]$Python = 'python',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [int]$Train = 400,
    [int]$Validation = 100,
    [int]$Test = 100,
    [int]$Bootstrap = 2000,
    [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:HF_HUB_DISABLE_XET = '1'
if (-not $SkipDownload) {
    & $Python scripts/download_pilot.py $Dataset --train $Train --val $Validation --test $Test
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
}
& $Python -m mmfnd.ablation.pilot --data "data/pilot/$Dataset" --out "outputs/pilot/$Dataset" --device $Device --encoder-batch-size 2
if ($LASTEXITCODE -ne 0) { throw 'Pilot training/evidence generation failed.' }
& $Python -m mmfnd.ablation.report --scores "outputs/pilot/$Dataset/scores.csv" --out "outputs/pilot/$Dataset/report" --bootstrap $Bootstrap
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
Write-Output "Results: $projectRoot/outputs/pilot/$Dataset/report"
