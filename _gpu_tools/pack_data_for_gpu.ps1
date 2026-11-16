# ============================================================================
# 打包"数据 + 工具"，跟代码分两条通道走。
#
#   powershell -ExecutionPolicy Bypass -File .\pack_data_for_gpu.ps1 -DryRun   # 只看会打什么
#   powershell -ExecutionPolicy Bypass -File .\pack_data_for_gpu.ps1           # 真的打包
#
# 产物：export\gpu_data_<日期>.zip  （约 150~175 MB）
#   hw3_imitation_learning\datasets\     遥操作录制数据（zarr，6 万+ 文件）
#   hw3_imitation_learning\checkpoints\  训练好的行为克隆权重（最好的 21 MB）
#   hw3_imitation_learning\hw3\keymap.json
#   hw2_robot_control_mdps\logs\         PPO 训练的 checkpoint
#   slides\                              课件 PDF（不需要可加 -NoSlides）
#   hw3_reference\                       专家演示视频（不需要可加 -NoReference）
#   _gpu_tools\                          目标机要用的 setup_gpu_env.sh + 手册 + env-lock
#
# 在目标机上的用法：把 zip 解压到**代码仓库根目录**，路径会自动归位。
#   例：~/work/ethz-gpu-work/  ← zip 解压到这里，于是 datasets/ 出现在 hw3_.../datasets/
# ============================================================================
[CmdletBinding()]
param(
  [string]$Destination = '',
  [switch]$DryRun,
  [switch]$NoReference,
  [switch]$NoSlides
)

$ErrorActionPreference = 'Stop'
$src    = $PSScriptRoot
if (-not $src) { $src = (Get-Location).Path }
$course = Join-Path $src 'ethz-robot-learning-2026'
if (-not (Test-Path $course)) { throw "找不到课程目录: $course" }

$stamp = Get-Date -Format 'yyyy-MM-dd'
if (-not $Destination) { $Destination = Join-Path $src 'export' }
$stage = Join-Path $Destination 'gpu_data_stage'
$zip   = Join-Path $Destination "gpu_data_$stamp.zip"

# ---- 清单：From(源) -> Rel(在 zip 里的相对路径) ----
$items = @(
  @{ From = "$course\hw3_imitation_learning\datasets";   Rel = 'hw3_imitation_learning\datasets';   Name = 'hw3 遥操作数据集' }
  @{ From = "$course\hw3_imitation_learning\checkpoints";Rel = 'hw3_imitation_learning\checkpoints';Name = 'hw3 训练权重' }
  @{ From = "$course\hw2_robot_control_mdps\logs";        Rel = 'hw2_robot_control_mdps\logs';       Name = 'hw2 PPO checkpoint' }
  @{ From = "$course\hw3_imitation_learning\hw3\keymap.json"; Rel = 'hw3_imitation_learning\hw3\keymap.json'; Name = 'hw3 键位表'; File = $true }
)
if (-not $NoSlides)    { $items += @{ From = "$course\slides"; Rel = 'slides'; Name = '课件 PDF（35 MB）' } }
if (-not $NoReference) { $items += @{ From = "$src\hw3_reference"; Rel = 'hw3_reference'; Name = '专家演示视频' } }

# ---- 目标机工具：从工作区根拷进 zip ----
$tools = @(
  "$src\setup_gpu_env.sh",
  "$src\GPU迁移_操作手册.md",
  "$src\MIGRATE_TO_GPU.md"
)

function Get-Size($p) {
  if (-not (Test-Path -LiteralPath $p)) { return $null }
  $s = Get-ChildItem -LiteralPath $p -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
  [pscustomobject]@{ MB = [math]::Round($s.Sum / 1MB, 1); Files = $s.Count }
}

Write-Host "源      : $src"
Write-Host "目标 zip: $zip"
Write-Host ''

# ---- DryRun ----
if ($DryRun) {
  $total = 0.0; $totalFiles = 0
  Write-Host '--- DRY RUN：以下是会被打包的内容 ---'
  foreach ($it in $items) {
    $m = Get-Size $it.From
    if ($null -eq $m) { Write-Host ('  {0,-34} (不存在，跳过)' -f $it.Rel); continue }
    Write-Host ('  {0,-34} {1,8:N1} MB  {2,6} 个文件   {3}' -f $it.Rel, $m.MB, $m.Files, $it.Name)
    $total += $m.MB; $totalFiles += $m.Files
  }
  foreach ($t in $tools) {
    if (Test-Path $t) { Write-Host ('  {0,-34} {1,8:N1} MB' -f ('_gpu_tools\' + (Split-Path $t -Leaf)), ((Get-Item $t).Length / 1MB)) }
    else { Write-Host ('  {0,-34} (缺失!)' -f $t) }
  }
  Write-Host ''
  Write-Host ('合计：{0} 个文件，约 {1:N0} MB（zip 后约 {2:N0} MB，zarr 小文件多、几乎压不动）' -f $totalFiles, $total, ($total * 1.12))
  Write-Host '(dry run：什么都没复制)'
  exit 0
}

# ---- 真打包 ----
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$manifest = New-Object System.Collections.Generic.List[string]
$manifest.Add("gpu data bundle  $stamp")
$manifest.Add("source: $src")
$manifest.Add('')

foreach ($it in $items) {
  if (-not (Test-Path -LiteralPath $it.From)) { Write-Host "  跳过（不存在）: $($it.From)"; continue }
  $dest = Join-Path $stage $it.Rel
  if ($it.File) {
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $it.From -Destination $dest -Force
  } else {
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    & robocopy $it.From $dest /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy 失败 ($LASTEXITCODE): $($it.From)" }
  }
  $m = Get-Size $dest
  Write-Host ('  packed {0,-34} {1,8:N1} MB  {2} 文件' -f $it.Rel, $m.MB, $m.Files)
  $manifest.Add(('{0,-34} {1,8:N1} MB  {2} files' -f $it.Rel, $m.MB, $m.Files))
}

$toolDir = Join-Path $stage '_gpu_tools'
New-Item -ItemType Directory -Path $toolDir -Force | Out-Null
foreach ($t in $tools) { if (Test-Path $t) { Copy-Item -LiteralPath $t -Destination $toolDir -Force } }
if (Test-Path "$src\env-lock") {
  Copy-Item -LiteralPath "$src\env-lock" -Destination (Join-Path $toolDir 'env-lock') -Recurse -Force
}
$manifest.Add('')
$manifest.Add('目标机重建步骤：')
$manifest.Add('  1) git clone <你的仓库> && git checkout my-work')
$manifest.Add('  2) unzip gpu_data_*.zip -d .        # 解压到仓库根，路径自动归位')
$manifest.Add('  3) bash _gpu_tools/setup_gpu_env.sh')
$manifest.Add('  4) 读 _gpu_tools/GPU迁移_操作手册.md 的"验收"一节')
Set-Content -Path (Join-Path $stage 'MANIFEST.txt') -Value $manifest -Encoding UTF8

$m = Get-Size $stage
Write-Host ('staged: {0} 个文件，{1:N1} MB' -f $m.Files, $m.MB)

if (Test-Path $zip) { Remove-Item $zip -Force }
$base = $zip -replace '\.zip$', ''
& python -c "import shutil,sys; shutil.make_archive(sys.argv[1],'zip',sys.argv[2])" $base $stage
if ($LASTEXITCODE -ne 0) { throw "make_archive 失败 ($LASTEXITCODE)" }

$z = Get-Item $zip
Write-Host ''
Write-Host ('zip : {0}' -f $z.FullName)
Write-Host ('大小: {0:N1} MB' -f ($z.Length / 1MB))
Write-Host '下一步：把 zip 传到目标机（U 盘 / 网盘 / scp / rsync），然后按手册第 3 步操作。'
