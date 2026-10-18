# 使用项目虚拟环境，在 PowerShell 直接清点与核对，无需浏览器。
[CmdletBinding()]
param(
    [string]$图片,
    [string]$结果,
    [string]$人工清单,
    [switch]$确认人工清点,
    [switch]$包含背景,
    [double]$置信度 = 0.25,
    [double]$去重阈值 = 0.45,
    [string]$输出目录
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw '未找到项目虚拟环境，请先按 README 准备环境。'
}
if (-not $图片 -and -not $结果) {
    $图片 = Read-Host '请输入桌面照片的完整路径（无需外层引号）'
}
if ($图片 -and $结果) { throw '图片分析和离线评估只能选择一个。' }
$arguments = @('-B', (Join-Path $PSScriptRoot 'compare_image.py'))
if ($结果) {
    if (-not $人工清单 -or -not $确认人工清点) { throw '离线核对需要 -人工清单 和 -确认人工清点。' }
    $arguments += @('--evaluate-result', $结果, '--truth', $人工清单, '--confirm-truth')
} else {
    if ($人工清单 -or $确认人工清点) { throw '先分析图片，生成对应的人工清点模板，再运行离线核对。' }
    $arguments += @($图片, '--confidence', $置信度.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--iou', $去重阈值.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($包含背景) { $arguments += '--include-background' }
if ($输出目录) { $arguments += @('--output', $输出目录) }
& $projectPython @arguments
if ($LASTEXITCODE -ne 0) {
    if ($LASTEXITCODE -eq 2) { Write-Warning '部分或全部方案未完成，已有结果已保留，请查看输出中的失败原因。'; exit 2 }
    Write-Warning '操作未完成，请查看上方中文提示。'
    exit 1
}
