param(
    [string]$Python = 'D:/codex/envs/resnet-cifar10/python.exe',
    [switch]$PushResults,
    [string]$GitProxy = ''
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$experimentBranch = (git branch --show-current).Trim()
if ($experimentBranch -eq 'main') { throw 'Use an experiment branch, not main.' }
$resultDir = 'results/seed1'
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
try {
    & $Python -u -m scripts.train_comparison --output $resultDir --resume
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
    $status = Get-Content "$resultDir/status.json" -Raw | ConvertFrom-Json
    if (-not $status.complete) { throw 'Four-model comparison is incomplete.' }
    if ($PushResults) {
        if ((git branch --show-current).Trim() -ne $experimentBranch) {
            throw 'Branch changed during training; leave results local for review.'
        }
        git add -- $resultDir
        if ($LASTEXITCODE -ne 0) { throw 'Cannot stage experiment results.' }
        git diff --cached --quiet -- $resultDir
        if ($LASTEXITCODE -eq 1) {
            git commit --only -m '[2260031896] 李霄汉 - Record measured four-model CIFAR-10 comparison' -- $resultDir
            if ($LASTEXITCODE -ne 0) { throw 'Cannot commit experiment results.' }
        }
        if ($GitProxy) {
            git -c "http.proxy=$GitProxy" push origin $experimentBranch
        } else {
            git push origin $experimentBranch
        }
        if ($LASTEXITCODE -ne 0) { throw 'Results saved locally; GitHub push failed.' }
    }
    Write-Output 'Four-model comparison completed successfully.'
} catch {
    $_ | Out-String | Set-Content "$resultDir/runner-error.txt"
    throw
}
