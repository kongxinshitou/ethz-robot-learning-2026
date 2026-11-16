"""
6 自由度机械臂逆运动学（IK）演示
--------------------------------
结构：6R + 球形手腕（Puma560 型 DH 参数），末端带 80 mm 工具长度。
内容：
  1) 正运动学 FK 与几何雅可比 J
  2) 数值 IK：阻尼最小二乘（DLS / Levenberg-Marquardt 特例）
  3) 多解：随机重启 + 解去重
  4) 反例：只给末端位置(x,y,z) 时，姿态不确定 -> 无穷多解
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------------- 1. 模型 ----------------
# 标准 DH: T_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
# (d, a, alpha) 单位 m / rad
LINKS = [
    (0.0000,  0.0000,  np.pi / 2),
    (0.0000,  0.4318,  0.0),
    (0.1501,  0.0203, -np.pi / 2),
    (0.4318,  0.0000,  np.pi / 2),
    (0.0000,  0.0000, -np.pi / 2),
    (0.0000,  0.0000,  0.0),
]
D_TOOL = 0.08                      # 法兰到工具末端 (沿 z6)
LIMITS = np.deg2rad([165, 165, 165, 165, 165, 165])


def dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk(q, with_frames=False):
    """返回末端 4x4 齐次矩阵；with_frames 时额外返回各连杆坐标系原点/轴。"""
    T = np.eye(4)
    frames = []
    for qi, (d, a, alpha) in zip(q, LINKS):
        frames.append(T.copy())          # 关节 i 的参考系 {i-1}
        T = T @ dh(qi, d, a, alpha)
    T_tool = T @ np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, D_TOOL],
                           [0, 0, 0, 1]], float)
    return (T_tool, frames) if with_frames else T_tool


def jacobian(q):
    """几何雅可比 6x6：[线速度; 角速度]，全部回转关节。"""
    T_tool, frames = fk(q, with_frames=True)
    p_e = T_tool[:3, 3]
    J = np.zeros((6, 6))
    for i, T in enumerate(frames):
        z = T[:3, 2]
        o = T[:3, 3]
        J[:3, i] = np.cross(z, p_e - o)
        J[3:, i] = z
    return J


# ---------------- 2. 姿态误差 ----------------
def rot_log(R):
    """SO(3) 对数映射：返回旋转向量（轴*角）。"""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    return th / (2.0 * np.sin(th)) * v


def pose_error(T_cur, T_des):
    """6 维位姿误差：[位置误差; 姿态误差(旋转向量)]。"""
    return np.concatenate([T_des[:3, 3] - T_cur[:3, 3],
                           rot_log(T_des[:3, :3] @ T_cur[:3, :3].T)])


# ---------------- 3. 数值 IK ----------------
def ik_dls(T_des, q0, max_iter=300, tol=1e-9, lam=1e-4, dq_max=0.25):
    """阻尼最小二乘迭代：dq = J^T (J J^T + lam*I)^-1 * e"""
    q = np.clip(np.asarray(q0, float), -LIMITS, LIMITS).copy()
    hist = []
    for _ in range(max_iter):
        T = fk(q)
        e = pose_error(T, T_des)
        hist.append(np.linalg.norm(e))
        if hist[-1] < tol:
            return q, True, hist
        J = jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        n = np.max(np.abs(dq))
        if n > dq_max:                 # 限幅，避免大步跳出工作区
            dq *= dq_max / n
        q = np.clip(q + dq, -LIMITS, LIMITS)
    T = fk(q)
    return q, np.linalg.norm(pose_error(T, T_des)) < 1e-6, hist


def report(tag, q, T_des, h):
    T = fk(q)
    ep = np.linalg.norm(T_des[:3, 3] - T[:3, 3]) * 1000
    er = np.rad2deg(np.linalg.norm(rot_log(T_des[:3, :3] @ T[:3, :3].T)))
    print(f"{tag}: 迭代 {len(h):3d} 次 | 位置误差 {ep:8.2e} mm | 姿态误差 {er:8.2e} deg")
    print("    q =", np.rad2deg(q), "deg")


# ---------------- 4. 实验 ----------------
rng = np.random.default_rng(0)

print("=" * 74)
print("实验 1：已知完整位姿 (位置+姿态)，用 DLS 数值迭代求 6 个关节角")
print("=" * 74)
q_true = np.deg2rad([20, -35, 40, 15, 55, -30])
T_des = fk(q_true)
# 只把位姿告诉求解器：位置(mm) + 旋转矩阵
print("目标位置 (mm):", np.round(T_des[:3, 3] * 1000, 2))
print("目标旋转矩阵 R :\n", np.round(T_des[:3, :3], 4))
q0 = q_true + rng.normal(0, 0.35, 6)          # 任意初值（偏 20° 左右）
q_sol, ok, h = ik_dls(T_des, q0)
report("求解结果", q_sol, T_des, h)
print("真值 q      =", np.rad2deg(q_true), "deg")
print("解与真值差  =", np.rad2deg((q_sol - q_true + np.pi) % (2 * np.pi) - np.pi), "deg")

print()
print("=" * 74)
print("实验 2：多解 —— 同一目标位姿，300 个随机初值能得到几个不同的关节角组合")
print("=" * 74)
sols = []
n_try = 300
for k in range(n_try):
    q0 = rng.uniform(-np.pi * 0.9, np.pi * 0.9, 6)
    q_s, ok, h = ik_dls(T_des, q0)
    if ok:
        qn = np.rad2deg((q_s + np.pi) % (2 * np.pi) - np.pi)
        if not any(np.max(np.abs(qn - s)) < 1e-3 for s in sols):
            sols.append(qn)
print(f"从 {n_try} 个随机初值出发，共找到 {len(sols)} 组互不相同的关节角（都精确到达同一位姿）：")
for i, s in enumerate(sols, 1):
    print(f"  解{i}: {s} deg   (最大关节角 {np.max(np.abs(s)):.1f}°)")

print()
print("=" * 74)
print("实验 3：只给末端位置 (x,y,z) 会怎样")
print("=" * 74)
q_a = np.deg2rad([10, -40, 50, 30, 40, 0])
p_des = fk(q_a)[:3, 3]


def ik_position_only(p_des, q0, lam=1e-4, iters=500):
    q = np.clip(np.asarray(q0, float), -LIMITS, LIMITS).copy()
    for _ in range(iters):
        T = fk(q)
        e = p_des - T[:3, 3]
        if np.linalg.norm(e) < 1e-10:
            break
        Jv = jacobian(q)[:3]
        dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + lam * np.eye(3), e)
        q = np.clip(q + dq, -LIMITS, LIMITS)
    return q


q_b = ik_position_only(p_des, np.deg2rad([10, 10, 10, 10, 10, 10]))
q_c = ik_position_only(p_des, np.deg2rad([60, -60, 90, -60, 30, 45]))
for tag, qq in [("解A", q_a), ("解B", q_b), ("解C", q_c)]:
    T = fk(qq)
    print(f"{tag}: 位置误差 {np.linalg.norm(p_des - T[:3, 3])*1000:7.2e} mm, "
          f"q = {np.round(np.rad2deg(qq), 1)} deg")
Rb = fk(q_b)[:3, :3]
print("解A 与 解B 的位置差 =",
      f"{np.linalg.norm(fk(q_a)[:3,3]-fk(q_b)[:3,3])*1000:.2e} mm",
      " 姿态夹角 =",
      f"{np.rad2deg(np.arccos(np.clip((np.trace(fk(q_a)[:3,:3] @ Rb.T)-1)/2,-1,1))):.1f} deg")
