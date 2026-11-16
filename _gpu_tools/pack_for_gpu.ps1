# Pack the study workspace for moving to a GPU machine.
# Drops the 3.3 GB of Windows virtualenvs and the 1.9 GB uv cache.
#
#   powershell -ExecutionPolicy Bypass -File .\pack_for_gpu.ps1            # build the zip
#   powershell -ExecutionPolicy Bypass -File .\pack_for_gpu.ps1 -DryRun    # just list + size
#
[CmdletBinding()]
param(
  [string]$Destination = '',
  [string]$ZipName = 'ethz-robot-learning-gpu.zip',
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
if (-not $src) { $src = (Get-Location).Path }
if (-not $Destination) { $Destination = Join-Path $src 'export' }
$stage = Join-Path $Destination 'stage'
$zip = Join-Path $Destination $ZipName

$excludeDirs = @('.venv','__pycache__','.mplcache','.uv-cache','.tmp','.ipynb_checkpoints','node_modules','export','downloads')
$excludeFiles = @('*.pyc','*.pyo','*证件照*','id-photo-*')

Write-Host "source : $src"
Write-Host "stage  : $stage"
Write-Host ("exclude: " + ($excludeDirs -join ', '))

if ($DryRun) {
  $pat = ($excludeDirs | ForEach-Object { [regex]::Escape('\' + $_ + '\') }) -join '|'
  $files = Get-ChildItem -LiteralPath $src -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
      $_.FullName -notmatch $pat -and
      $_.Extension -ne '.pyc' -and
      $_.Name -notlike '*证件照*' -and
      $_.Name -notlike 'id-photo-*'
    }
  $m = $files | Measure-Object -Property Length -Sum
  Write-Host ''
  Write-Host ('--- DRY RUN: {0} files, {1:N1} MB would be packed ---' -f $m.Count, ($m.Sum/1MB))
  $files | Group-Object {
      $rel = $_.FullName.Substring($src.Length).TrimStart('\')
      if ($rel -match '\\') { ($rel -split '\\')[0] } else { '(files in root)' }
    } | Sort-Object Name | ForEach-Object {
      $s = ($_.Group | Measure-Object -Property Length -Sum).Sum
      Write-Host ('  {0,-28} {1,8:N1} MB  ({2} files)' -f $_.Name, ($s/1MB), $_.Count)
    }
  Write-Host ''
  Write-Host '(dry run: nothing copied, nothing compressed)'
  exit 0
}

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$roboArgs = @($src, $stage, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:1', '/W:1', '/XD') + $excludeDirs + @('/XF') + $excludeFiles

& robocopy @roboArgs | Out-Null
$rc = $LASTEXITCODE
if ($rc -ge 8) { throw "robocopy failed with exit code $rc" }

$m = Get-ChildItem $stage -Recurse -File | Measure-Object -Property Length -Sum
Write-Host ('packed files: {0}, {1:N1} MB' -f $m.Count, ($m.Sum/1MB))

if (Test-Path $zip) { Remove-Item $zip -Force }
$py = Get-Command python -ErrorAction SilentlyContinue
if ($py) {
  $base = $zip -replace '\.zip$', ''
  & python -c "import shutil,sys; shutil.make_archive(sys.argv[1],'zip',sys.argv[2])" $base $stage
  if ($LASTEXITCODE -ne 0) { throw "python make_archive failed ($LASTEXITCODE)" }
} else {
  Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
}
$z = Get-Item $zip
Write-Host ('zip: ' + $z.FullName + '  (' + [math]::Round($z.Length/1MB,1) + ' MB)')