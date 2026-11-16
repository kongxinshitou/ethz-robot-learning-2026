"""
6 关节平面（二维）臂：已知末端点 (x, y)，求 6 个关节角
------------------------------------------------------
核心事实：2 个方程、6 个未知数 -> 冗余 4 维，解不唯一。
必须再补 4 个条件。下面给三种补法，各自对应一段代码：

  A. 分层法（最快、最常用）：固定其中 4 个关节 -> 剩下 2 个有闭式解（2R，两分支）
  B. 约束优化：min 关节运动量 s.t. 末端点 = 目标、关节在行程内（SLSQP）
  C. 一参数解族：固定前 3 个关节，让第 6 关节自由扫掠 -> 得到一族精确解（画成扇形）

输出：figure_planar.png
"""
import numpy as np
from scipy.optimize import minimize
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(precision=1, suppress=True)

L = np.array([8.0, 7.2, 6.8, 6.2, 5.8, 5.2])       # 6 节臂长 (m)
N = len(L)
Q_MIN, Q_MAX = np.deg2rad(0.0), np.deg2rad(170.0)  # 关节行程（此例取“只能向上抬”）
P_DES = np.array([30.0, 20.0])                     # 已知末端点 (m)


def fk(q):
    """q 为各节与 +x 轴夹角（绝对角）。返回基座 + 6 个关节点。"""
    pts = [np.zeros(2)]
    for li, qi in zip(L, q):
        pts.append(pts[-1] + li * np.array([np.cos(qi), np.sin(qi)]))
    return np.array(pts)


def jac(q):
    return np.vstack([-L * np.sin(q), L * np.cos(q)])


def solve_2r(v, l1, l2):
    """从当前点出发走向量 v，求 2 根杆的绝对角；返回 0~2 组解。"""
    r = np.linalg.norm(v)
    if r > l1 + l2 or r < abs(l1 - l2):
        return []
    c = np.clip((r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1)
    s = np.sqrt(max(0.0, 1 - c * c))
    phi = np.arctan2(v[1], v[0])
    out = []
    for sg in (+1, -1):
        th1 = phi - np.arctan2(l2 * sg * s, l1 + l2 * c)
        out.append((th1, th1 + sg * np.arctan2(s, c)))
    return out


# =============== A. 分层法：固定 4 个关节，解剩下 2 个 ===============
print("=" * 70)
print("A. 分层法：q1~q4 取当前姿态，q5/q6 用 2R 闭式解")
print("=" * 70)
q_cur = np.deg2rad([50, 30, 15, 25, 0, 0])
q_fix = q_cur.copy()
p4 = fk(q_fix)[4]                       # 第 5 节起点
v = P_DES - p4
print(f"  固定 q1~q4 = {np.round(np.rad2deg(q_fix[:4]), 1)} deg")
print(f"  剩余向量 v = {np.round(v, 2)} m, |v| = {np.linalg.norm(v):.2f} m "
      f"(第5+6节可伸 11.0 m)")
for i, (t5, t6) in enumerate(solve_2r(v, L[4], L[5]), 1):
    qq = q_fix.copy()
    qq[4], qq[5] = t5, t6
    err = np.linalg.norm(fk(qq)[-1] - P_DES) * 1000
    print(f"  分支{i}: q5={np.rad2deg(t5):6.2f}°, q6={np.rad2deg(t6):6.2f}° "
          f"-> 末端误差 {err:.1e} mm")

# =============== B. 约束优化 ===============
print()
print("=" * 70)
print("B. 约束优化：min ||q - q0||^2  s.t. 末端点 = 目标, 0 <= qi <= 170 deg")
print("=" * 70)
seeds = {
    "seed A": np.deg2rad([45, 45, 45, 45, 45, 45]),
    "seed B": np.deg2rad([70, 50, 45, 40, 35, 30]),
    "seed C": np.deg2rad([60, 30, 20, 20, 20, 20]),
}
opt_sols = {}
for name, q0 in seeds.items():
    r = minimize(lambda q: np.sum((q - q0) ** 2), q0,
                 constraints=[{"type": "eq", "fun": lambda q: fk(q)[-1] - P_DES}],
                 bounds=[(Q_MIN, Q_MAX)] * N, method="SLSQP",
                 options={"maxiter": 400, "ftol": 1e-12})
    q = r.x
    err = np.linalg.norm(fk(q)[-1] - P_DES) * 1000
    opt_sols[name] = q
    print(f"  {name}: 收敛={r.success}, 误差={err:.1e} mm, "
          f"q = {np.round(np.rad2deg(q), 1)} deg")
print(f"  目标点 {P_DES} m；三组解的关节角最大差 "
      f"{np.rad2deg(np.max(np.abs(opt_sols['seed A'] - opt_sols['seed B']))):.1f}°")

# =============== C. 一参数解族 ===============
print()
print("=" * 70)
print("C. 固定 q1~q3，扫掠 q6 -> 每个 q6 都对应一组精确解（解族）")
print("=" * 70)
q_base = np.deg2rad([50, 30, 15])
p3 = fk(np.concatenate([q_base, [0, 0, 0]]))[3]
fam = []
for q6 in np.deg2rad(np.arange(-70, 105, 5)):
    w = P_DES - p3 - L[5] * np.array([np.cos(q6), np.sin(q6)])
    for t4, t5 in solve_2r(w, L[3], L[4]):
        q = np.concatenate([q_base, [t4, t5, q6]])
        if np.all(q >= Q_MIN - 1e-9) and np.all(q <= Q_MAX):
            fam.append(q)
if fam:
    print(f"  q6 从 -70° 扫到 100°，得到 {len(fam)} 组可行解，"
          f"q4 范围 {np.rad2deg(min(f[3] for f in fam)):.1f}° ~ "
          f"{np.rad2deg(max(f[3] for f in fam)):.1f}°")
    print("  抽 3 组验证末端点：")
    for k in (0, len(fam) // 2, len(fam) - 1):
        print(f"    q6={np.rad2deg(fam[k][5]):6.1f}° -> 末端 "
              f"{np.round(fk(fam[k])[-1], 6)} , 误差 "
              f"{np.linalg.norm(fk(fam[k])[-1]-P_DES)*1000:.1e} mm")

# =============== 画图 ===============
W, H = 1200, 780
xmin, xmax, ymin, ymax = -4, 42, -3, 33
sc = min(W / (xmax - xmin), H / (ymax - ymin))
ox, oy = (W - sc * (xmax - xmin)) / 2, (H + sc * (ymax - ymin)) / 2


def to_px(p):
    return (ox + (p[0] - xmin) * sc, oy - (p[1] - ymin) * sc)


img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)
try:
    f_big, f_small = ImageFont.load_default(20), ImageFont.load_default(15)
except TypeError:
    f_big = f_small = ImageFont.load_default()

for gx in range(0, 41, 5):
    dr.line([to_px([gx, ymin]), to_px([gx, ymax])], fill=(235, 235, 235))
for gy in range(0, 31, 5):
    dr.line([to_px([xmin, gy]), to_px([xmax, gy])], fill=(235, 235, 235))
dr.line([to_px([xmin, 0]), to_px([xmax, 0])], fill=(70, 70, 70), width=2)

for q in fam:                                        # 解族
    dr.line([to_px(p) for p in fk(q)], fill=(205, 214, 227), width=3)

colors = [(28, 110, 180), (214, 96, 34), (36, 150, 60)]
for (name, q), c in zip(opt_sols.items(), colors):   # 三个优化解
    pts = [to_px(p) for p in fk(q)]
    dr.line(pts, fill=c, width=6, joint="curve")
    for p in pts[1:-1]:
        rr = 5
        dr.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr],
                   fill="white", outline=c, width=3)

tp = to_px(P_DES)
dr.ellipse([tp[0] - 13, tp[1] - 13, tp[0] + 13, tp[1] + 13],
           outline=(200, 30, 30), width=4)
dr.ellipse([tp[0] - 4, tp[1] - 4, tp[0] + 4, tp[1] + 4], fill=(200, 30, 30))
bp = to_px([0, 0])
dr.rectangle([bp[0] - 15, bp[1] - 15, bp[0] + 15, bp[1] + 15], fill=(40, 40, 40))

dr.text((22, 14), "6 joints, 2D task: 2 equations, 6 unknowns", fill=(20, 20, 20), font=f_big)
dr.text((22, 44), "grey = one-parameter family of exact solutions", fill=(120, 120, 120), font=f_small)
dr.text((22, 66), "colored = 3 constrained-optimization results", fill=(120, 120, 120), font=f_small)
dr.text((22, 96), f"tip = ({P_DES[0]:.0f}, {P_DES[1]:.0f}) m", fill=(200, 30, 30), font=f_small)
for i, name in enumerate(opt_sols):
    dr.text((22, 126 + 22 * i), name, fill=colors[i], font=f_small)

img.save("figure_planar.png")
print("\n已保存 figure_planar.png")
