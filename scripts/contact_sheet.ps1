# Compose a labelled contact sheet from per-frame review_srgb.png renders.
# Usage: powershell -File scripts\contact_sheet.ps1 [-RunDir output\testroll_full] [-Out <path>]
param(
    [string]$RunDir = "output\testroll_full",
    [string]$Out = ""
)

Add-Type -AssemblyName System.Drawing

if (-not $Out) { $Out = Join-Path $RunDir "contact_sheet.png" }
$frameDirs = Get-ChildItem -Directory $RunDir | Where-Object { Test-Path (Join-Path $_.FullName "review_srgb.png") } | Sort-Object Name
if (-not $frameDirs) { Write-Error "no review_srgb.png renders under $RunDir"; exit 1 }

$summaryPath = Join-Path $RunDir "summary.txt"
$summary = @()
if (Test-Path $summaryPath) { $summary = Get-Content $summaryPath }

$cols = 4
$rows = [math]::Ceiling($frameDirs.Count / $cols)
$tileW = 440; $imgH = 290; $labelH = 20; $pad = 6
$tileH = $imgH + $labelH
$sheetW = $cols * ($tileW + $pad) + $pad
$sheetH = $rows * ($tileH + $pad) + $pad

$sheet = New-Object System.Drawing.Bitmap($sheetW, $sheetH)
$g = [System.Drawing.Graphics]::FromImage($sheet)
$g.Clear([System.Drawing.Color]::FromArgb(24, 24, 24))
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$font = New-Object System.Drawing.Font("Consolas", 8)
$brush = [System.Drawing.Brushes]::White

for ($i = 0; $i -lt $frameDirs.Count; $i++) {
    $name = $frameDirs[$i].Name
    $imgPath = Join-Path $frameDirs[$i].FullName "review_srgb.png"
    $img = [System.Drawing.Image]::FromFile((Resolve-Path $imgPath))
    $col = $i % $cols
    $row = [math]::Floor($i / $cols)
    $x = $pad + $col * ($tileW + $pad)
    $y = $pad + $row * ($tileH + $pad)
    $scale = [math]::Min($tileW / $img.Width, $imgH / $img.Height)
    $w = [int]($img.Width * $scale); $h = [int]($img.Height * $scale)
    $ox = $x + [int](($tileW - $w) / 2); $oy = $y + [int](($imgH - $h) / 2)
    $g.DrawImage($img, $ox, $oy, $w, $h)
    $img.Dispose()

    $label = "RAW_$name"
    $line = $summary | Where-Object { $_ -match "RAW_$name " } | Select-Object -First 1
    if ($line -and $line -match "trust=(\S+).*span=(\S+) ev=(\S+) mid=(\S+)") {
        $label = "$name $($Matches[1]) span=$($Matches[2]) ev=$($Matches[3]) mid=$($Matches[4])"
    }
    $g.DrawString([string]$label, $font, $brush, $x, $y + $imgH + 3)
}

$g.Dispose()
$sheet.Save((Join-Path (Get-Location) $Out), [System.Drawing.Imaging.ImageFormat]::Png)
$sheet.Dispose()
Write-Output "contact sheet: $Out ($($frameDirs.Count) frames, ${sheetW}x${sheetH})"
