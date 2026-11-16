#!/usr/bin/env bash
# ============================================================================
# 在"有显卡的机器"上重建 4 个作业的虚拟环境。
#
#   ❌ 绝对不要从 Windows 机器复制 .venv 过来（3.4 GB + 跨平台二进制 + CPU 版 torch）
#   ✅ 用这个脚本重建，10~25 分钟，视网速
#
# 用法（在解压出的课程目录里执行，也就是能看到 hw1_pytorch_tutorial 的那一层）：
#   bash setup_gpu_env.sh
#   bash setup_gpu_env.sh /path/to/ethz-robot-learning-2026      # 显式指定目录
#   CUDA_INDEX=https://download.pytorch.org/whl/cu124 bash setup_gpu_env.sh   # 锁 CUDA 版本
#
# 说明：
#   * Linux 上 PyPI 的 torch wheel 自带 CUDA runtime，一般不需要指定 CUDA_INDEX；
#     只有当你必须匹配某个特定 CUDA 版本（如 cu121）时才指定。
#   * hw2、hw3 都提供同名 so101_gym 包，所以必须 4 个独立 venv，不能合并。
#   * hw3 的 student_eval/*.so 只有 linux / darwin 版 —— 在 Linux 上反而能用了。
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-3.12}"
CUDA_INDEX="${CUDA_INDEX:-}"

# ---- 定位课程根目录（兼容两种布局：脚本在课程根里 / 脚本在课程根的上一层）----
resolve_root() {
  local cand
  for cand in "${1:-}" "$(pwd)" "$HERE" "$HERE/ethz-robot-learning-2026" "$HERE/.."; do
    if [ -n "$cand" ] && [ -d "$cand/hw1_pytorch_tutorial" ]; then
      (cd "$cand" && pwd)
      return 0
    fi
  done
  return 1
}
if ! ROOT="$(resolve_root "${1:-}")"; then
  echo "ERROR: 找不到课程目录（需要有 hw1_pytorch_tutorial 子目录）" >&2
  echo "用法: bash setup_gpu_env.sh [path/to/ethz-robot-learning-2026]" >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || {
  echo "需要 uv，先装:  curl -LsSf https://astral.sh/uv/install.sh | sh   (或 pip install uv)" >&2
  exit 1
}
export UV_LINK_MODE=copy

ve()    { uv venv --python "$PY" "$ROOT/$1/.venv" >/dev/null; }
pybin() { echo "$ROOT/$1/.venv/bin/python"; }
pipi()  { local hw="$1"; shift; uv pip install --python "$(pybin "$hw")" "$@"; }
fallback_pth() {
  local hw="$1"
  local sp="$ROOT/$hw/.venv/lib/python$PY/site-packages"
  mkdir -p "$sp"
  echo "$ROOT/$hw" > "$sp/_hw_local.pth"
  echo "  fallback: 已写 $sp/_hw_local.pth（等价于 pip install -e .）"
}

echo "== 课程目录 : $ROOT"
echo "== Python   : $PY   (hw3 的 .so 是 cpython-312，必须是 3.12)"
echo "== CUDA idx : ${CUDA_INDEX:-<不指定，用 PyPI 默认>}"
echo

echo "[1/4] hw1: torch + torchvision + jupyterlab"
ve hw1_pytorch_tutorial
if [ -n "$CUDA_INDEX" ]; then
  pipi hw1_pytorch_tutorial torch torchvision --index-url "$CUDA_INDEX"
else
  pipi hw1_pytorch_tutorial torch torchvision
fi
pipi hw1_pytorch_tutorial jupyterlab ipykernel

echo "[2/4] hw2: mujoco / gymnasium / stable-baselines3"
ve hw2_robot_control_mdps
pipi hw2_robot_control_mdps -r "$ROOT/hw2_robot_control_mdps/requirements.txt"
pipi hw2_robot_control_mdps -e "$ROOT/hw2_robot_control_mdps" || fallback_pth hw2_robot_control_mdps

echo "[3/4] hw3: torch / zarr / opencv / dm-control"
ve hw3_imitation_learning
if [ -n "$CUDA_INDEX" ]; then
  pipi hw3_imitation_learning torch torchvision --index-url "$CUDA_INDEX"
fi
pipi hw3_imitation_learning -e "$ROOT/hw3_imitation_learning" || fallback_pth hw3_imitation_learning
# 工作区根目录那些 hw3_*.py 画图脚本会 import matplotlib，env-lock 里没有它，补上
pipi hw3_imitation_learning matplotlib

echo "[4/4] hw4: torch / gymnasium / moviepy / pygame"
ve hw4_reinforcement_learning
if [ -n "$CUDA_INDEX" ]; then
  pipi hw4_reinforcement_learning torch --index-url "$CUDA_INDEX"
fi
pipi hw4_reinforcement_learning -r "$ROOT/hw4_reinforcement_learning/requirements.txt"

echo
echo "===== 冒烟测试 ====="
"$(pybin hw1_pytorch_tutorial)" -c "import torch,torchvision;print('hw1 torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda)"
"$(pybin hw2_robot_control_mdps)" -c "import mujoco,gymnasium,stable_baselines3 as s;print('hw2 mujoco',mujoco.__version__,'sb3',s.__version__)"
"$(pybin hw3_imitation_learning)" -c "import torch,zarr,cv2,dm_control;print('hw3 zarr',zarr.__version__,'cuda',torch.cuda.is_available())"
"$(pybin hw4_reinforcement_learning)" -c "import gymnasium,torch;print('hw4 gym',gymnasium.__version__,'cuda',torch.cuda.is_available())"

echo
echo "===== 作业级自检（失败不影响环境，只提示）====="
if [ -f "$ROOT/hw2_robot_control_mdps/check_ex1.py" ]; then
  ( cd "$ROOT/hw2_robot_control_mdps" && "$(pybin hw2_robot_control_mdps)" check_ex1.py ) || echo "  ! hw2 check_ex1.py 未通过，看输出"
fi
( cd "$ROOT/hw4_reinforcement_learning" && "$(pybin hw4_reinforcement_learning)" -c "
from envs.grid_world import CliffWalkingEnv
from envs.cartpole_wrapper import CartPoleWrapper
g=CliffWalkingEnv(); c=CartPoleWrapper(seed=0); s=c.reset(); c.step(c.sample_action()); print('hw4 smoke A OK', g.n_states)
" ) || echo "  ! hw4 smoke A 失败"

echo
echo "===== 下一步 ====="
echo "cd $ROOT/hw2_robot_control_mdps && .venv/bin/python check_ex1.py"
echo "# 无桌面的服务器才需要："
echo "export MUJOCO_GL=egl"
echo "# sudo apt install -y libegl1 libgl1 libglvnd0 libglew2.2  （无头渲染依赖）"
