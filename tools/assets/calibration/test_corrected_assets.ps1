[CmdletBinding()]
param(
    [ValidateSet('Coarse', 'Fine', 'Manual')]
    [string]$Mode = 'Coarse',
    [ValidateSet('1st', '2nd', '3rd', 'b1', 'b2', 'b3', 'b4')]
    [string]$Floor = '1st',
    [ValidateRange(1, 32768)]
    [int]$ReferenceWidth = 2560,
    [ValidateRange(1, 32768)]
    [int]$ReferenceHeight = 1440,
    [double]$Center = [double]::NaN,
    [double]$Scale = [double]::NaN,
    [double]$Minimum = 0.5,
    [double]$Maximum = 4.5,
    [double]$Step = 0.05,
    [string]$Python,
    [string]$AssetRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
if (-not $AssetRoot) {
    $AssetRoot = Join-Path $repoRoot 'workspace/asset_build/rebuilt_assets'
}
$AssetRoot = [System.IO.Path]::GetFullPath($AssetRoot)
$clean = Join-Path $AssetRoot "a_bringer_of_doom/hard/clean/layout_01/layout_01_$Floor.png"
$shot = Join-Path $AssetRoot "captures/layout_01_${Floor}_min.png"

try {
    foreach ($inputPath in @($clean, $shot)) {
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Missing input PNG: $inputPath"
        }
    }
    if (-not $Python) {
        $candidates = @(
            (Join-Path $repoRoot '.venv/Scripts/python.exe'),
            'E:/IdentityV_Cryptic_Notes_Assistant/.venv/Scripts/python.exe'
        )
        $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (-not $Python) {
            throw 'Existing Python environment not found. Supply -Python <python.exe>.'
        }
    }
    $output = Join-Path $AssetRoot "scale_trials/layout_01/$Floor/$($Mode.ToLowerInvariant())"
    $toolArgs = @(
        (Join-Path $PSScriptRoot 'calibrate_minimum_scale.py'),
        '--clean', $clean, '--screenshot', $shot,
        '--reference-width', "$ReferenceWidth", '--reference-height', "$ReferenceHeight",
        '--output', $output
    )
    function NumberText([double]$Value) {
        return $Value.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
    }
    if ($Mode -eq 'Manual') {
        if ([double]::IsNaN($Scale) -or [double]::IsInfinity($Scale) -or $Scale -le 0) {
            throw 'Manual mode requires a finite positive -Scale value.'
        }
        $toolArgs += @('--scale', (NumberText $Scale))
    } else {
        if ($Mode -eq 'Fine') {
            if ([double]::IsNaN($Center)) {
                $previous = Join-Path $AssetRoot "scale_trials/layout_01/$Floor/coarse/candidate.json"
                if (-not (Test-Path -LiteralPath $previous -PathType Leaf)) {
                    throw 'Run Coarse first, or supply -Center for Fine mode.'
                }
                $candidate = Get-Content -LiteralPath $previous -Raw | ConvertFrom-Json
                if ($candidate.reference_size[0] -ne $ReferenceWidth -or $candidate.reference_size[1] -ne $ReferenceHeight) {
                    throw 'Reference resolution changed. Run Coarse again or supply -Center.'
                }
                $Center = [double]$candidate.scale
            }
            if ([double]::IsInfinity($Center) -or $Center -le 0) {
                throw 'Fine center must be a finite positive scale.'
            }
            $Minimum = [Math]::Max(0.001, $Center - 0.05)
            $Maximum = $Center + 0.05
            $Step = 0.001
        }
        $toolArgs += @('--min', (NumberText $Minimum), '--max', (NumberText $Maximum), '--step', (NumberText $Step))
    }
    & $Python @toolArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "Preview:   $(Join-Path $output 'preview.png')"
    Write-Host "Candidate: $(Join-Path $output 'candidate.json')"
    Write-Host 'Review the preview, then use minimum_scale from candidate.json. Outputs in this mode are replaced on rerun.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
