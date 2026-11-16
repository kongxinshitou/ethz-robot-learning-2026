# -*- coding: utf-8 -*-
"""
PID 入门教学仿真：一阶对象 + 纯滞后，画阶跃响应曲线。
用法: python pid_demo.py
输出: fig1_progression.png  fig2_pitfalls.png
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_R = r"C:\Windows\Fonts\msyh.ttc"
FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"
SS = 2  # 超采样倍数，抗锯齿


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, int(size * SS))


# ---------------------------------------------------------------- 仿真核心
def simulate(kp=0.0, ki=0.0, kd=0.0, *, T=2.0, L=0.6, dt=0.005, t_end=30.0,
             sat=None, anti_windup=False, noise=0.0, d_tau=0.0, d_on_meas=True,
             open_loop_u=None, ref_time=1.0, ref_high=1.0, seed=3):
    """一阶对象 dy/dt = (u - y)/T，带纯滞后 L。PID 为位置式，输出限幅 sat。"""
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    r = np.where(t >= ref_time, ref_high, 0.0)
    y = np.zeros(n + 1)
    u = np.zeros(n + 1)
    I = 0.0
    e_prev = 0.0
    d_state = 0.0
    rng = np.random.default_rng(seed)
    nz = rng.normal(0.0, noise, n + 1) if noise > 0 else np.zeros(n + 1)

    nbuf = max(1, int(round(L / dt)))
    ubuf = [0.0] * nbuf

    for k in range(n):
        if open_loop_u is not None:
            uk = open_loop_u
        else:
            ym = y[k] + nz[k]
            e = r[k] - ym
            P = kp * e
            if ki:
                I += ki * e * dt
            if kd:
                src = -ym if d_on_meas else e      # 测量值微分，避免设定值突变冲击
                raw = (src - e_prev) / dt
                if d_tau > 0:                      # 一阶低通滤波
                    d_state += dt / (d_tau + dt) * (raw - d_state)
                    D = kd * d_state
                else:
                    D = kd * raw
                e_prev = src
            else:
                D = 0.0
            uk = P + I + D
            if sat is not None:
                u_sat = min(max(uk, sat[0]), sat[1])
                if anti_windup and uk != u_sat:    # 条件积分抗饱和
                    if (uk > u_sat and e > 0) or (uk < u_sat and e < 0):
                        I -= ki * e * dt
                uk = u_sat
            if anti_windup:                        # 积分限幅兜底
                I = min(max(I, -2.0), 2.0)
        u[k] = uk
        u_app = ubuf.pop(0)
        ubuf.append(uk)
        y[k + 1] = y[k] + dt / T * (u_app - y[k])

    u[n] = u[n - 1]
    return t, r, y, u


def metrics(t, r, y, ref_time=1.0):
    m = t >= ref_time
    tt, yy, rr = t[m], y[m], r[m]
    target = rr[-1]
    peak = np.max(yy) if len(yy) else 0.0
    overshoot = max(0.0, (peak - target) / target * 100) if target else 0.0
    tail = yy[int(0.9 * len(yy)):]
    ss_err = np.mean(tail) - target
    band = 0.02 * abs(target) if target else 0.02
    outside = np.where(np.abs(yy - target) > band)[0]
    settle = tt[outside[-1]] - ref_time if len(outside) else 0.0
    early = yy[int(0.8 * len(yy)):]
    ripple = np.max(early) - np.min(early)
    return dict(overshoot=overshoot, ss_err=ss_err, settle=settle, ripple=ripple)


# ---------------------------------------------------------------- 画图工具
COL = {
    "blue":   (31, 119, 180),
    "orange": (255, 127, 14),
    "green":  (44, 160, 44),
    "red":    (214, 39, 40),
    "purple": (148, 103, 189),
    "gray":   (120, 120, 120),
    "ink":    (45, 45, 45),
}


def P(v):
    return int(round(v * SS))


def dashed(draw, p0, p1, color, width=2, dash=8, gap=6):
    x0, y0 = p0
    x1, y1 = p1
    ln = np.hypot(x1 - x0, y1 - y0)
    if ln == 0:
        return
    ux, uy = (x1 - x0) / ln, (y1 - y0) / ln
    s = 0.0
    while s < ln:
        e = min(s + dash * SS, ln)
        draw.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)],
                  fill=color, width=P(width))
        s = e + gap * SS


def rot_text(img, xy, text, fnt, color=(90, 90, 90)):
    tmp = Image.new("RGBA", (P(220), P(40)), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    td.text((0, 0), text, font=fnt, fill=color)
    bbox = tmp.getbbox()
    if bbox is None:
        return
    tmp = tmp.crop(bbox).rotate(90, expand=True)
    img.paste(tmp, xy, tmp)


def draw_panel(img, draw, box, title, series, xlim, ylim, refs=(), xticks=None,
               yticks=None, ylabel="输出 y", xlabel="时间 t / s", note=""):
    x0, y0, x1, y1 = [P(v) for v in box]

    def X(v):
        return x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)

    def Y(v):
        return y1 - (v - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)

    # 网格
    if xticks is None:
        xticks = np.arange(np.ceil(xlim[0]), xlim[1] + 1e-9, 5)
    if yticks is None:
        yticks = np.arange(np.ceil(ylim[0] / 0.5) * 0.5, ylim[1] + 1e-9, 0.5)
    for xt in xticks:
        draw.line([(X(xt), y0), (X(xt), y1)], fill=(232, 232, 232), width=P(1))
    for yt in yticks:
        draw.line([(x0, Y(yt)), (x1, Y(yt))], fill=(232, 232, 232), width=P(1))

    # 参考线与数据曲线画在独立图层，再裁剪进面板，避免超范围数据画到面板外
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)

    for (_, col, xs, ys, *rest) in refs:
        if not isinstance(xs, (list, np.ndarray)):
            continue
        pts = [(X(a), Y(b)) for a, b in zip(xs, ys)]
        for i in range(len(pts) - 1):
            if rest and rest[0]:
                dashed(ld, pts[i], pts[i + 1], COL[col], 2)
            else:
                ld.line([pts[i], pts[i + 1]], fill=COL[col], width=P(2))

    for (_, col, xs, ys, *rest) in series:
        w = rest[0] if rest else 2.5
        pts = [(X(a), Y(b)) for a, b in zip(xs, ys)]
        ld.line(pts, fill=COL[col], width=P(w), joint="curve")

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
    blank = Image.new("RGBA", img.size, (0, 0, 0, 0))
    comp = Image.composite(layer, blank, mask)
    img.paste(comp, (0, 0), comp)

    # 坐标轴
    draw.line([(x0, y0), (x0, y1)], fill=(60, 60, 60), width=P(2))
    draw.line([(x0, y1), (x1, y1)], fill=(60, 60, 60), width=P(2))

    # 刻度文字
    f_tick = font(11)
    for xt in xticks:
        s = f"{xt:g}"
        draw.text((X(xt) - draw.textlength(s, font=f_tick) / 2, y1 + P(6)),
                  s, fill=(80, 80, 80), font=f_tick)
    for yt in yticks:
        s = f"{yt:g}"
        draw.text((x0 - P(7) - draw.textlength(s, font=f_tick), Y(yt) - P(9)),
                  s, fill=(80, 80, 80), font=f_tick)

    # 标题 / 轴名 / 注记
    draw.text((x0, y0 - P(46)), title, fill=(20, 20, 20), font=font(14, bold=True))
    if note:
        draw.text((x0 + P(2), y0 - P(24)), note, fill=(176, 61, 61), font=font(10.5))
    draw.text((x1 - draw.textlength(xlabel, font=f_tick), y1 + P(22)),
              xlabel, fill=(90, 90, 90), font=f_tick)
    rot_text(img, (x0 - P(40), y0 + P(4)), ylabel, font(11))

    # 图例（右上，白底避免压线）
    if series:
        f_lg = font(11)
        tw_max = max(draw.textlength(lab, font=f_lg) for lab, *_ in series)
        box_w = tw_max + P(46)
        box_h = len(series) * P(17) + P(12)
        bx1, by0 = x1 - P(6), y0 + P(4)
        draw.rectangle([bx1 - box_w, by0, bx1, by0 + box_h],
                       fill=(255, 255, 255), outline=(214, 214, 214))
        for i, (lab, col, *_rest) in enumerate(series):
            yy = by0 + P(6) + i * P(17)
            tw = draw.textlength(lab, font=f_lg)
            draw.line([(bx1 - P(6) - tw - P(28), yy + P(7)),
                       (bx1 - P(6) - tw - P(8), yy + P(7))],
                      fill=COL[col], width=P(3))
            draw.text((bx1 - P(6) - tw, yy), lab, fill=(50, 50, 50), font=f_lg)


def label_chars(s):
    return list(s)


def decimate(xs, ys, n=1600):
    if len(xs) <= n:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return xs[idx], ys[idx]


def build_fig1(path):
    W, H = 1180, 910
    img = Image.new("RGB", (P(W), P(H)), "white")
    d = ImageDraw.Draw(img)

    xlim = (0, 25)
    ylim = (0, 1.8)
    ref_x = [0, 1, 1, 25]
    ref_y = [0, 0, 1, 1]

    panels = [(60, 135, 570, 425), (640, 135, 1150, 425),
              (60, 570, 570, 860), (640, 570, 1150, 860)]

    d.text((P(60), P(42)), "图 1  PID 是怎么一步步长出来的（对象：一阶惯性 + 0.6 s 纯滞后，目标：输出从 0 升到 1）",
           fill=(20, 20, 20), font=font(16, bold=True))

    # ① 开环 vs 闭环
    t, r, y_ol, _ = simulate(open_loop_u=0.6, t_end=25)
    _, _, y_cl, _ = simulate(kp=2.0, t_end=25)
    draw_panel(img, d, panels[0], "① 为什么要反馈：开环 vs 闭环",
               [("开环（u=0.6 固定）", "orange", *decimate(t, y_ol)),
                ("闭环 P（Kp=2）", "blue", *decimate(t, y_cl))],
               xlim, ylim, refs=[("目标", "ink", ref_x, ref_y, True)],
               note="两条线都停在 1 以下 → 有余差（稳态误差）")

    # ② 只有 P
    t, r, y1, _ = simulate(kp=1.0, t_end=25)
    _, _, y3, _ = simulate(kp=3.0, t_end=25)
    _, _, y55, _ = simulate(kp=5.5, t_end=25)
    draw_panel(img, d, panels[1], "② 只有 P：增益越大越快，也越容易振荡",
               [("Kp=1（慢、余差大）", "blue", *decimate(t, np.clip(y1, 0, 1.8))),
                ("Kp=3（快、仍有差）", "green", *decimate(t, np.clip(y3, 0, 1.8))),
                ("Kp=5.5（振荡）", "red", *decimate(t, np.clip(y55, 0, 1.8)))],
               xlim, ylim, refs=[("目标", "ink", ref_x, ref_y, True)],
               note="余差始终在；Kp 越大余差越小，但振荡越凶")

    # ③ 加 I
    t, r, y80, _ = simulate(kp=3.0, ki=0.0, t_end=25)
    _, _, y83, _ = simulate(kp=3.0, ki=0.8, t_end=25)
    _, _, y86, _ = simulate(kp=3.0, ki=2.0, t_end=25)
    draw_panel(img, d, panels[2], "③ 加上 I：余差被消掉，代价是超调与振荡",
               [("Ki=0（纯 P，有差）", "gray", *decimate(t, np.clip(y80, 0, 1.8))),
                ("Ki=0.8（刚好补上）", "green", *decimate(t, np.clip(y83, 0, 1.8))),
                ("Ki=2（积太快，大超调）", "red", *decimate(t, np.clip(y86, 0, 1.8)))],
               xlim, ylim, refs=[("目标", "ink", ref_x, ref_y, True)],
               note="看末端：加了 I 的曲线都爬到了 1 → 余差消失")

    # ④ 加 D
    t, r, yD0, _ = simulate(kp=3.0, ki=0.8, kd=0.0, t_end=25)
    _, _, yD1, _ = simulate(kp=3.0, ki=0.8, kd=1.0, t_end=25)
    _, _, yD2, _ = simulate(kp=3.0, ki=0.8, kd=1.7, t_end=25)
    draw_panel(img, d, panels[3], "④ 加上 D：提前刹车，把超调压下去",
               [("Kd=0（超调大）", "gray", *decimate(t, np.clip(yD0, 0, 1.8))),
                ("Kd=1（阻尼刚好）", "green", *decimate(t, np.clip(yD1, 0, 1.8))),
                ("Kd=1.7（过头，反而抖）", "purple", *decimate(t, np.clip(yD2, 0, 1.8)))],
               xlim, ylim, refs=[("目标", "ink", ref_x, ref_y, True)],
               note="D 看的是“变化速度”：快到目标时自动收力")

    img = img.resize((W, H), Image.LANCZOS)
    img.save(path, quality=95)
    return path


def build_fig2(path):
    W, H = 1180, 500
    img = Image.new("RGB", (P(W), P(H)), "white")
    d = ImageDraw.Draw(img)
    d.text((P(60), P(36)), "图 2  两个工程陷阱：积分饱和，以及 D 项放大噪声",
           fill=(20, 20, 20), font=font(16, bold=True))
    ref_x = [0, 1, 1, 25]
    ref_y = [0, 0, 1, 1]

    # 左：积分饱和
    t, r, yw, _ = simulate(kp=3.0, ki=4.0, sat=(0, 1.6), t_end=25)
    _, _, yaw, _ = simulate(kp=3.0, ki=4.0, sat=(0, 1.6), anti_windup=True, t_end=25)
    draw_panel(img, d, (60, 125, 570, 425), "⑤ 执行器饱和：积分饱和 vs 抗饱和",
               [("无抗饱和（超调 48%）", "red", *decimate(t, yw)),
                ("有抗饱和（超调 9%）", "blue", *decimate(t, yaw))],
               (0, 25), (0, 1.9),
               refs=[("目标", "ink", ref_x, ref_y, True)],
               note="阀门已开满到 1.6，误差还在积分 → 退饱和时冲过头")

    # 右：噪声下的微分（稳态段放大）
    t, r, _, u_raw = simulate(kp=3.0, ki=1.0, kd=0.1, noise=0.004, t_end=25)
    _, _, _, u_f = simulate(kp=3.0, ki=1.0, kd=0.1, noise=0.004, d_tau=0.15, t_end=25)
    draw_panel(img, d, (640, 125, 1150, 425), "⑥ 稳态段放大：D 项把测量噪声放大",
               [("D 直接微分（锯齿抖动）", "red", *decimate(t, u_raw), 1.2),
                ("D 加低通滤波（平滑）", "blue", *decimate(t, u_f), 2.5)],
               (12, 20), (0.5, 1.55), ylabel="控制量 u",
               xticks=np.arange(12, 21, 2), yticks=np.arange(0.6, 1.55, 0.2),
               note="噪声被 1/Δt 放大后直接进了执行器")

    img = img.resize((W, H), Image.LANCZOS)
    img.save(path, quality=95)
    return path


def arrow(d, p0, p1, color=(70, 70, 70), w=2.5, head=9):
    x0, y0 = p0
    x1, y1 = p1
    d.line([(x0, y0), (x1, y1)], fill=color, width=P(w))
    ln = np.hypot(x1 - x0, y1 - y0)
    if ln < 1:
        return
    ux, uy = (x1 - x0) / ln, (y1 - y0) / ln
    hs = P(head)
    bx, by = x1 - ux * hs, y1 - uy * hs
    d.polygon([(x1, y1), (bx - uy * hs * 0.45, by + ux * hs * 0.45),
               (bx + uy * hs * 0.45, by - ux * hs * 0.45)], fill=color)


def build_fig3(path):
    W, H = 1180, 400
    img = Image.new("RGB", (P(W), P(H)), "white")
    d = ImageDraw.Draw(img)
    d.text((P(40), P(28)), "图 3  闭环 PID 的结构：目标 → 误差 → 控制器 → 对象 → 测回来再比较",
           fill=(20, 20, 20), font=font(16, bold=True))

    cy = 205
    f_lab = font(14, bold=True)
    f_small = font(12)

    # 目标 r
    d.text((P(36), P(cy) - P(10)), "目标 r(t)", font=f_lab, fill=(20, 20, 20))
    arrow(d, (P(126), P(cy)), (P(176), P(cy)))

    # 求和点
    cx = P(200)
    d.ellipse([cx - P(22), P(cy) - P(22), cx + P(22), P(cy) + P(22)],
              outline=(70, 70, 70), width=P(2.5), fill=(255, 255, 255))
    d.text((cx - P(7), P(cy) - P(12)), "Σ", font=font(16), fill=(70, 70, 70))
    d.text((cx + P(14), P(cy) + P(12)), "−", font=font(18, bold=True), fill=(70, 70, 70))

    # PID 框
    arrow(d, (cx + P(22), P(cy)), (P(268), P(cy)))
    d.rounded_rectangle([P(268), P(cy) - P(58), P(520), P(cy) + P(58)],
                        radius=P(10), fill=(233, 242, 250), outline=(31, 119, 180), width=P(2.5))
    d.text((P(286), P(cy) - P(40)), "PID 控制器", font=font(15, bold=True), fill=(20, 60, 110))
    d.text((P(286), P(cy) - P(14)), "u = Kp·e + Ki∫e dt + Kd·de/dt", font=f_small, fill=(40, 70, 110))
    d.text((P(286), P(cy) + P(10)), "（三个旋钮：比例 / 积分 / 微分）", font=f_small, fill=(90, 110, 130))

    # 对象框
    arrow(d, (P(520), P(cy)), (P(568), P(cy)))
    d.rounded_rectangle([P(568), P(cy) - P(58), P(820), P(cy) + P(58)],
                        radius=P(10), fill=(254, 244, 228), outline=(255, 127, 14), width=P(2.5))
    d.text((P(586), P(cy) - P(40)), "被控对象（执行器 + 物体）", font=font(13.5, bold=True), fill=(120, 70, 10))
    d.text((P(586), P(cy) - P(14)), "电机 / 加热器 / 阀门 / 水泵…", font=f_small, fill=(140, 95, 40))
    d.text((P(586), P(cy) + P(10)), "有自己的惯性、延迟、非线性", font=f_small, fill=(140, 95, 40))

    # 输出
    arrow(d, (P(820), P(cy)), (P(898), P(cy)))
    d.text((P(902), P(cy) - P(10)), "输出 y(t)", font=f_lab, fill=(20, 20, 20))
    d.text((P(902), P(cy) + P(14)), "实际测量值", font=f_small, fill=(90, 90, 90))

    # 反馈支路
    fx = P(860)
    d.ellipse([fx - P(4), P(cy) - P(4), fx + P(4), P(cy) + P(4)], fill=(70, 70, 70))
    fy = P(cy + 128)
    d.line([(fx, P(cy)), (fx, fy)], fill=(70, 70, 70), width=P(2.5))
    d.line([(fx, fy), (cx, fy)], fill=(70, 70, 70), width=P(2.5))
    arrow(d, (cx, fy), (cx, P(cy) + P(22)))
    d.text((P(470), fy - P(22)), "测量反馈（传感器）", font=f_small, fill=(90, 90, 90))

    # 关键标注
    d.text((P(196), P(cy) - P(84)), "误差 e = r − y", font=f_small, fill=(176, 61, 61))
    d.text((P(500), P(cy) - P(84)), "控制量 u", font=f_small, fill=(176, 61, 61))
    d.text((P(36), P(cy) + P(74)),
           "核心：不猜“该给多少”，只看“现在差多少” → 用 P（差多少）、I（欠多少）、D（变多快）决定加多少力。",
           font=font(12.5), fill=(60, 60, 60))

    img = img.resize((W, H), Image.LANCZOS)
    img.save(path, quality=95)
    return path


def report():
    rows = [
        ("开环 u=0.6", dict(open_loop_u=0.6)),
        ("P    Kp=1", dict(kp=1.0)),
        ("P    Kp=3", dict(kp=3.0)),
        ("P    Kp=5.5", dict(kp=5.5)),
        ("PI   Kp=3 Ki=0.8", dict(kp=3.0, ki=0.8)),
        ("PI   Kp=3 Ki=2.0", dict(kp=3.0, ki=2.0)),
        ("PID  Kp=3 Ki=0.8 Kd=1.0", dict(kp=3.0, ki=0.8, kd=1.0)),
        ("PID  Kp=3 Ki=0.8 Kd=1.7", dict(kp=3.0, ki=0.8, kd=1.7)),
    ]
    print(f"{'方案':<24}{'超调%':>9}{'稳态误差':>11}{'调节时间s':>11}{'尾部纹波':>10}")
    print("-" * 68)
    for name, kw in rows:
        t, r, y, _ = simulate(t_end=25, **kw)
        m = metrics(t, r, y)
        print(f"{name:<24}{m['overshoot']:>9.1f}{m['ss_err']:>11.3f}"
              f"{m['settle']:>11.2f}{m['ripple']:>10.3f}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    report()
    print()
    print("saved:", build_fig1("fig1_progression.png"))
    print("saved:", build_fig2("fig2_pitfalls.png"))
    print("saved:", build_fig3("fig3_loop.png"))
