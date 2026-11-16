# ============================================================================
# 在"有 NVIDIA 显卡的 Windows 机器"上重建 4 个作业的虚拟环境（PowerShell 原生版）
#
#   ❌ 不要从旧机器复制 .venv：Windows 的 torch 是 CPU 版，跨机器复制也用不上 GPU
#   ✅ 跑这个脚本，10~25 分钟（看网速），最后自动验收
#
# 用法（在仓库根目录，也就是能看到 hw1_pytorch_tutorial 的那一层）：
#   powershell -ExecutionPolicy Bypass -File .\_gpu_tools\setup_gpu_env.ps1
#   ... -Root D:\work\ethz-robot-learning-2026      # 显式指定仓库根
#   ... -TorchBackend cu124                        # 锁 CUDA 版本（默认 auto）
#   ... -TorchBackend cpu                          # 没有显卡时（慢慢跑也行）
#
# 说明：
#   * Windows 上 PyPI 的 torch 是 CPU-only 版，必须用 --torch-backend 从 PyTorch 官方
#     的 CUDA 索引装；auto = 让 uv 根据你的显卡驱动自动挑 cuXXX。
#   * hw2、hw3 都提供同名 so101_gym 包 → 必须 4 个独立 venv，不能合并。
#   * hw3 的 student_eval\*.so 只有 linux/darwin 版 → Windows 上本地评分跑不了，
#     用 scripts\eval.py，最终分数以 Gradescope 为准。
# ============================================================================
[CmdletBinding()]
param(
  [string]$Root = '',
  [ValidateSet('auto', 'cpu', 'cu118', 'cu121', 'cu124', 'cu126', 'cu128')]
  [string]$TorchBackend = 'auto',
  [string]$Py = '3.12',
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# ---------- 1. 定位仓库根 ----------
function Resolve-Root {
  param([string]$Hint)
  $cands = New-Object System.Collections.Generic.List[string]
  if ($Hint) { $cands.Add($Hint) }
  $cands.Add((Get-Location).Path)
  $here = $PSScriptRoot
  $cands.Add($here)
  $cands.Add((Split-Path $here -Parent))
  $cands.Add((Join-Path $here 'ethz-robot-learning-2026'))
  foreach ($c in $cands) {
    if ($c -and (Test-Path (Join-Path $c 'hw1_pytorch_tutorial'))) { return (Resolve-Path $c).Path }
  }
  return $null
}
$Root = Resolve-Root $Root
if (-not $Root) {
  Write-Host 'ERROR: 找不到仓库根目录（需要有 hw1_pytorch_tutorial 子目录）' -ForegroundColor Red
  Write-Host '用法: powershell -ExecutionPolicy Bypass -File .\_gpu_tools\setup_gpu_env.ps1 -Root <path>' -ForegroundColor Yellow
  exit 1
}

# ---------- 2. 环境自检 ----------
Write-Host '===== 环境自检 =====' -ForegroundColor Cyan
Write-Host "仓库根 : $Root"
Write-Host "Python : $Py"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $smi = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null) -join ' / '
  Write-Host "显卡   : $smi"
  $cudaLine = (& nvidia-smi 2>$null | Select-String -Pattern 'CUDA Version' | Select-Object -First 1)
  if ($cudaLine) { Write-Host "         $($cudaLine.Line.Trim())" }
} else {
  Write-Host '⚠ 没找到 nvidia-smi：要么没装驱动，要么这台机器没有 NVIDIA 显卡。' -ForegroundColor Yellow
  Write-Host '  装完 CUDA 驱动再来；只想先跑通流程可以用 -TorchBackend cpu' -ForegroundColor Yellow
}

$script:UvExe = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $script:UvExe) {
  Write-Host ''
  Write-Host 'ERROR: 需要 uv。任选一种安装：' -ForegroundColor Red
  Write-Host '   winget install --id=astral-sh.uv -e'
  Write-Host '   pip install uv'
  exit 1
}
Write-Host "uv     : $($script:UvExe)"
$env:UV_LINK_MODE = 'copy'

# ---------- 3. 工具函数（显式数组传参，避免 --python 被当成函数参数名）----------
function Invoke-Uv {
  param([string[]]$ArgList)
  if ($DryRun) { Write-Host ("    [dry] uv " + ($ArgList -join ' ')) -ForegroundColor DarkGray; return }
  & $script:UvExe @ArgList
  if ($LASTEXITCODE -ne 0) { throw "uv 命令失败 (exit $LASTEXITCODE): uv $($ArgList -join ' ')" }
}
function Venv-Python { param([string]$Hw) Join-Path $Root "$Hw\.venv\Scripts\python.exe" }

function New-HwVenv {
  param([string]$Hw)
  Invoke-Uv @('venv', '--python', $Py, (Join-Path $Root "$Hw\.venv"))
}
function Install-Hw {
  param([string]$Hw, [string[]]$Packages)
  $argv = @('pip', 'install', '--python', (Venv-Python $Hw)) + $Packages
  Invoke-Uv $argv
}
function Install-HwEditable {
  param([string]$Hw)
  try {
    Invoke-Uv @('pip', 'install', '--python', (Venv-Python $Hw), '-e', (Join-Path $Root $Hw), '--torch-backend', $TorchBackend)
  } catch {
    Write-Host '  pip install -e . 失败，用 .pth 兜底' -ForegroundColor Yellow
    $sp = Join-Path $Root "$Hw\.venv\Lib\site-packages"
    New-Item -ItemType Directory -Path $sp -Force | Out-Null
    Set-Content -Path (Join-Path $sp '_hw_local.pth') -Value (Join-Path $Root $Hw) -Encoding ASCII
    Write-Host "  fallback: 已写 $sp\_hw_local.pth（等价于 pip install -e .）" -ForegroundColor Yellow
  }
}
function Test-Cuda {
  param([string]$Hw)
  $out = (& (Venv-Python $Hw) -c "import torch;print(torch.__version__, 'cuda=', torch.cuda.is_available(), torch.version.cuda)" 2>&1) -join ' '
  Write-Host ("  {0,-28} {1}" -f $Hw, $out)
  if ($out -notmatch 'cuda= True') { Write-Host '    ⚠ 这个环境没吃到 GPU' -ForegroundColor Yellow }
}

Write-Host ''
Write-Host "===== 开始重建（torch backend = $TorchBackend）=====" -ForegroundColor Cyan

# ---------- 4. 逐个作业 ----------
Write-Host '[1/4] hw1: torch + torchvision + jupyterlab' -ForegroundColor Green
New-HwVenv 'hw1_pytorch_tutorial'
Invoke-Uv @('pip', 'install', '--python', (Venv-Python 'hw1_pytorch_tutorial'), 'torch', 'torchvision', '--torch-backend', $TorchBackend)
Install-Hw 'hw1_pytorch_tutorial' @('jupyterlab', 'ipykernel')

Write-Host '[2/4] hw2: mujoco / gymnasium / stable-baselines3' -ForegroundColor Green
New-HwVenv 'hw2_robot_control_mdps'
Invoke-Uv @('pip', 'install', '--python', (Venv-Python 'hw2_robot_control_mdps'), '-r', (Join-Path $Root 'hw2_robot_control_mdps\requirements.txt'), '--torch-backend', $TorchBackend)
Install-HwEditable 'hw2_robot_control_mdps'

Write-Host '[3/4] hw3: torch / zarr / opencv / dm-control' -ForegroundColor Green
New-HwVenv 'hw3_imitation_learning'
Invoke-Uv @('pip', 'install', '--python', (Venv-Python 'hw3_imitation_learning'), 'torch', 'torchvision', '--torch-backend', $TorchBackend)
Install-HwEditable 'hw3_imitation_learning'
# 仓库根目录那些 hw3_*.py 画图脚本会 import matplotlib，env-lock 里没有
Install-Hw 'hw3_imitation_learning' @('matplotlib')

Write-Host '[4/4] hw4: torch / gymnasium / moviepy / pygame' -ForegroundColor Green
New-HwVenv 'hw4_reinforcement_learning'
Invoke-Uv @('pip', 'install', '--python', (Venv-Python 'hw4_reinforcement_learning'), 'torch', '--torch-backend', $TorchBackend)
Invoke-Uv @('pip', 'install', '--python', (Venv-Python 'hw4_reinforcement_learning'), '-r', (Join-Path $Root 'hw4_reinforcement_learning\requirements.txt'), '--torch-backend', $TorchBackend)

# ---------- 5. 验收 ----------
if (-not $DryRun) {
Write-Host ''
Write-Host '===== 冒烟测试（四行都必须是 cuda= True）=====' -ForegroundColor Cyan
foreach ($hw in 'hw1_pytorch_tutorial', 'hw2_robot_control_mdps', 'hw3_imitation_learning', 'hw4_reinforcement_learning') {
  Test-Cuda $hw
}

Write-Host ''
Write-Host '===== 作业级自检 =====' -ForegroundColor Cyan
Push-Location (Join-Path $Root 'hw2_robot_control_mdps')
$py2 = Venv-Python 'hw2_robot_control_mdps'
foreach ($chk in 'check_ex1.py', 'check_ex2.py', 'check_ex3.py') {
  if (Test-Path $chk) {
    Write-Host "  -> $chk"
    & $py2 $chk
    if ($LASTEXITCODE -ne 0) { Write-Host "  ⚠ $chk 未通过，看上面的输出" -ForegroundColor Yellow }
  }
}
Pop-Location

Push-Location (Join-Path $Root 'hw4_reinforcement_learning')
$py4 = Venv-Python 'hw4_reinforcement_learning'
Write-Host '  -> hw4 smoke A'
& $py4 -c "from envs.grid_world import CliffWalkingEnv; from envs.cartpole_wrapper import CartPoleWrapper; g=CliffWalkingEnv(); c=CartPoleWrapper(seed=0); s=c.reset(); c.step(c.sample_action()); print('hw4 smoke A OK', g.n_states)"
Write-Host '  -> hw4 smoke B'
& $py4 -c "from pathlib import Path; import numpy as np; from envs.so100_rl_env import SO100RLEnv; e=SO100RLEnv(xml_path=Path('assets/mujoco/so100_pos_ctrl.xml').resolve(), render_mode=None); o,_=e.reset(seed=0); e.step(np.zeros(e.action_dim, dtype=np.float32)); print('hw4 smoke B OK', o.shape)"
Pop-Location

}  # end of (-not $DryRun)
Write-Host ''
Write-Host '===== 数据是否到位 =====' -ForegroundColor Cyan
foreach ($p in 'hw3_imitation_learning\datasets', 'hw3_imitation_learning\checkpoints', 'hw2_robot_control_mdps\logs') {
  $full = Join-Path $Root $p
  if (Test-Path $full) {
    $s = Get-ChildItem $full -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum
    Write-Host ("  {0,-34} {1,8:N1} MB  {2} 文件" -f $p, ($s.Sum / 1MB), $s.Count)
  } else {
    Write-Host ("  {0,-34} 缺失 ← 数据 zip 还没解压" -f $p) -ForegroundColor Yellow
  }
}

Write-Host ''
Write-Host '===== 下一步 =====' -ForegroundColor Cyan
Write-Host "  cd $Root\hw2_robot_control_mdps"
Write-Host '  .venv\Scripts\python.exe scripts\interactive.py     # 弹 MuJoCo 窗口'
Write-Host "  cd $Root\hw3_imitation_learning"
Write-Host '  .venv\Scripts\python.exe scripts\train.py --help'
Write-Host ''
Write-Host '提示：hw3 的 student_eval\*.so 只有 linux/darwin 版，Windows 上跑不了本地评测（用 scripts\eval.py）。' -ForegroundColor Yellow
