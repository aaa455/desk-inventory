# 使用已有入口完成分析、查看和终端人工核对。
[CmdletBinding()]
param(
    [string]$image,
    [string]$result,
    [string]$truth,
    [switch]$confirmTruth,
    [switch]$show,
    [switch]$review,
    [switch]$openImage,
    [switch]$includeBackground,
    [double]$confidence = 0.25,
    [double]$iou = 0.45,
    [string]$output
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw '未找到项目虚拟环境，请先按 README 准备环境。'
}
if ($show -and $review) { throw '查看和人工核对只能选择一个。' }
if ($openImage -and -not $show) { throw '-OpenImage 需与 -Show 一起使用。' }
if ($image -and ($result -or $show -or $review)) { throw '图片分析不能与读取结果同时使用。' }
if (($show -or $review) -and ($truth -or $confirmTruth)) {
    throw '简化查看或核对不需要提供人工清单文件和确认开关。'
}
if (-not $image -and -not $result -and -not $show -and -not $review) {
    $image = Read-Host '请输入桌面照片的完整路径（可拖入文件）'
}
if (-not $show -and -not $review -and -not $result -and -not $image) {
    throw '照片路径不能为空。'
}
$arguments = @('-B', (Join-Path $PSScriptRoot 'compare_image.py'))
if ($show -or $review) {
    if ($show) { $arguments += '--show' } else { $arguments += '--review' }
    if ($result) { $arguments += @('--evaluate-result', $result) }
    if ($openImage) { $arguments += '--open-image' }
} elseif ($result) {
    if (-not $truth -or -not $confirmTruth) { throw '离线核对需要 -Truth 和 -ConfirmTruth。' }
    $arguments += @('--evaluate-result', $result, '--truth', $truth, '--confirm-truth')
} else {
    if ($truth -or $confirmTruth) { throw '先分析图片，再核对人工清单。' }
    $image = $image.Trim().Trim('"').Trim("'")
    $arguments += @($image, '--confidence', $confidence.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--iou', $iou.ToString([Globalization.CultureInfo]::InvariantCulture))
}
if ($includeBackground) { $arguments += '--include-background' }
if ($output) { $arguments += @('--output', $output) }
$env:PYTHONIOENCODING = 'utf-8'
& $projectPython @arguments
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    if ($exitCode -eq 2) { Write-Warning '部分或全部方案未完成，已有结果已保留，请查看输出中的失败原因。'; exit 2 }
    Write-Warning '操作未完成，请查看上方中文提示。'
    exit 1
}
