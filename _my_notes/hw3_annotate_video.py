"""Turn the scripted expert demo into a 0.5x slow-motion, keyframe-annotated video.

Reads a run directory produced by hw3_demo_grasp.py:
    expert_demo.mp4     (10 fps, three camera views)
    expert_demo.zarr    (state_ee / state_cube / action_gripper per control step)
    meta.json           (phase boundaries)

Writes:
    expert_demo_slow_annotated.mp4   (0.5x speed, phase banner + live readouts)
    expert_demo_keyframes.png        (keyframe cheat sheet)

Usage (run inside hw3_imitation_learning with its venv):
    .venv\\Scripts\\python.exe ..\\..\\hw3_annotate_video.py [run_dir] [--speed 0.5]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

# ── phase → (编号, 标题, 要点) ────────────────────────────────────────
PHASE_INFO: dict[str, tuple[str, str, str]] = {
    "open gripper": ("01", "张开夹爪", "先按 {open} 把夹爪开到最大 (1.75)，再开始移动"),
    "lift": ("02", "抬高到 z ≈ 0.20", "先升到高处再平移；低处平移会撞到障碍"),
    "around barrier (left)": (
        "03", "从左侧绕过障碍",
        "障碍占 x −0.265…−0.175，走 x ≈ −0.28 走廊，不要钻 12 cm 的缝",
    ),
    "align with cube": (
        "04", "对准方块正上方",
        "用 top 视角对齐 XY；EE 点比如方块中心左偏约 1.2 cm",
    ),
    "descend": ("05", "只降 Z 到 z ≈ 0.04", "夹爪全程张开；一次只按一个轴，按一下等一格"),
    "grasp": ("06", "分步闭合夹爪", "分多次小步按 {close}；合到方块两侧后停 0.4 s 再抬"),
    "lift cube": ("07", "抬 5–10 cm 验证抓稳", "方块跟着起来 = 抓住了；没跟手就按 {reset} 重置重来"),
    "carry around barrier": (
        "08", "绕回障碍左侧搬运",
        "保持 EE z ≈ 0.20（方块底部 0.15 > 墙顶 0.12）",
    ),
    "to bin": ("09", "移到箱子上方", "用 top 视角对准箱中心（略偏左约 2 cm 补偿开爪偏移）"),
    "insert into bin": ("10", "送进箱子再松手", "EE 降到 z ≈ 0.09，让方块进到箱壁以内；别从高处松"),
    "release": ("11", "分步张开释放", "分多步按 {open}；方块落到箱底（z < 0.04 即成功）"),
    "retreat": ("12", "抬臂退回", "按 {up} 退出，避免刮到箱沿"),
}

# filled in from the run's actual keymap before drawing
KEY_LABELS: dict[str, str] = {
    "open": "M", "close": "L", "up": "↑", "reset": "R", "record": "Space", "end": "Enter",
}

PHASE_COLORS = [
    (0, 120, 215), (0, 153, 102), (230, 126, 34), (155, 89, 182),
    (41, 128, 185), (192, 57, 43), (39, 174, 96), (243, 156, 18),
    (142, 68, 173), (22, 160, 133), (211, 84, 0), (127, 140, 141),
]

ACTION_TEXT = {
    "move_up": "上移 1 cm",
    "move_down": "下移 1 cm",
    "move_left": "左移 1 cm",
    "move_right": "右移 1 cm",
    "move_forward": "前移 1 cm",
    "move_backward": "后移 1 cm",
    "gripper_open": "张开夹爪 +0.10",
    "gripper_close": "闭合夹爪 -0.10",
    "noop": "停一拍（不按键）",
}


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    g = zarr.open_group(str(run_dir / "expert_demo.zarr"), mode="r")
    ee = np.asarray(g["data"]["state_ee"][:])
    cube = np.asarray(g["data"]["state_cube"][:])
    grip = np.asarray(g["data"]["action_gripper"][:]).reshape(-1)
    return meta, ee, cube, grip


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def draw_frame(
    frame: np.ndarray,
    step: int,
    n_steps: int,
    phase_idx: int,
    phases: list[dict],
    ee: np.ndarray,
    cube: np.ndarray,
    grip: float,
    speed: float,
    fonts,
    tick: int,
    key_info: dict | None = None,
) -> np.ndarray:
    title_font, detail_font, small_font, tiny_font = fonts
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size  # 1280 x 960

    phase = phases[phase_idx]
    num, title, _ = PHASE_INFO.get(phase["name"], ("", phase["name"], ""))
    tip = phase_tip(phase["name"])
    color = PHASE_COLORS[phase_idx % len(PHASE_COLORS)]

    # top banner (opaque: hides the burned-in camera labels)
    draw.rectangle((0, 0, w, 104), fill=(0, 0, 0, 255))
    draw.rectangle((0, 0, 8, 104), fill=(*color, 255))
    draw.text((26, 12), f"关键帧 {num} · {title}", font=title_font, fill=(255, 255, 255, 255))
    draw.text((26, 64), tip, font=detail_font, fill=(255, 214, 102, 255))

    # re-draw camera captions below the banner
    draw.text((26, 112), "left_wrist · 夹爪特写", font=tiny_font, fill=(170, 170, 170, 255))
    draw.text((w // 2 + 26, 112), "angle · 全局", font=tiny_font, fill=(170, 170, 170, 255))

    # speed badge
    badge = f"{speed:g}× 慢放"
    bb = draw.textbbox((0, 0), badge, font=small_font)
    bw, bh = bb[2] - bb[0] + 24, bb[3] - bb[1] + 16
    rounded_rect(draw, (w - bw - 20, 14, w - 20, 14 + bh), 8, (192, 57, 43, 235))
    draw.text((w - bw - 8, 14 + 8 - bb[1]), badge, font=small_font, fill=(255, 255, 255, 255))

    # keyboard panel (bottom-right black area)
    if key_info is not None:
        px0, py0, px1, py1 = 872, 498, 1256, 792
        rounded_rect(draw, (px0, py0, px1, py1), 14, (8, 10, 14, 205),
                     outline=(*color, 220), width=2)
        draw.text((px0 + 22, py0 + 14),
                  f"当前按键 · 共 {key_info['total']} 拍", font=tiny_font,
                  fill=(170, 175, 185, 255))
        kx0, ky0, kx1, ky1 = px0 + 22, py0 + 44, px0 + 220, py0 + 44 + 150
        rounded_rect(draw, (kx0, ky0, kx1, ky1), 16, (30, 34, 42, 255),
                     outline=(255, 255, 255, 235), width=3)
        key_txt = key_info["key"]
        kf = find_font(86 if len(key_txt) == 1 else 46, bold=True)
        kbb = draw.textbbox((0, 0), key_txt, font=kf)
        draw.text(
            ((kx0 + kx1 - (kbb[2] - kbb[0])) / 2, ky0 + (ky1 - ky0 - (kbb[3] - kbb[1])) / 2 - kbb[1]),
            key_txt, font=kf, fill=(255, 255, 255, 255),
        )
        draw.text((px0 + 240, py0 + 52), key_info["text"], font=small_font,
                  fill=(255, 214, 102, 255))
        draw.text((px0 + 240, py0 + 88), "下一拍", font=tiny_font, fill=(150, 155, 165, 255))
        draw.text((px0 + 240, py0 + 112), key_info["next_key"], font=small_font,
                  fill=(200, 205, 215, 255))

        # key stream (last 8 presses)
        draw.text((px0 + 22, py0 + 212), "按键流（最近 8 拍，左旧右新）", font=tiny_font,
                  fill=(150, 155, 165, 255))
        stream = key_info["stream"][-8:]
        cx = px0 + 22
        for k in stream:
            cw = 32
            rounded_rect(draw, (cx, py0 + 236, cx + cw, py0 + 236 + 40), 7,
                         (40, 44, 52, 255), outline=(120, 125, 135, 200), width=1)
            sf = find_font(22 if len(k) == 1 else 13, bold=True)
            kb = draw.textbbox((0, 0), k, font=sf)
            draw.text((cx + (cw - (kb[2] - kb[0])) / 2, py0 + 236 + 20 - (kb[3] - kb[1]) / 2 - kb[1]),
                      k, font=sf, fill=(255, 255, 255, 255))
            cx += cw + 5

    # bottom info bar
    draw.rectangle((0, h - 170, w, h), fill=(0, 0, 0, 180))
    rel = step - phase["start"] + 1
    rel_total = phase["end"] - phase["start"]
    jaw_state = "张开" if grip > 1.5 else ("夹紧" if grip < 0.15 else "中间")
    line1 = f"step {step + 1}/{n_steps}   阶段内 {rel}/{rel_total}"
    line2 = (
        f"EE z = {ee[step, 2]:.3f} m    cube z = {cube[step, 2]:.3f} m"
        f"    夹爪 = {grip:+.2f} ({jaw_state})"
    )
    draw.text((24, h - 158),
              f"阶段 {phase_idx + 1}/{len(phases)} · 原速 {step / 10.0:.1f}s",
              font=tiny_font, fill=(190, 190, 190, 255))
    draw.text((24, h - 122), line1, font=small_font, fill=(220, 220, 220, 255))
    draw.text((24, h - 86), line2, font=small_font, fill=(255, 255, 255, 255))

    # progress bar with phase segments
    bar_x0, bar_x1, bar_y = 24, w - 24, h - 34
    bar_h = 16
    draw.rectangle((bar_x0, bar_y, bar_x1, bar_y + bar_h), fill=(40, 40, 40, 255))
    for i, ph in enumerate(phases):
        x0 = bar_x0 + (bar_x1 - bar_x0) * ph["start"] / n_steps
        x1 = bar_x0 + (bar_x1 - bar_x0) * ph["end"] / n_steps
        c = PHASE_COLORS[i % len(PHASE_COLORS)]
        alpha = 255 if i < phase_idx else (150 if i == phase_idx else 80)
        draw.rectangle((int(x0), bar_y, int(x1) - 1, bar_y + bar_h), fill=(*c, alpha))
        if i == phase_idx:
            draw.rectangle((int(x0), bar_y - 3, int(x1) - 1, bar_y + bar_h + 3),
                           outline=(255, 255, 255, 255), width=3)
    x = bar_x0 + (bar_x1 - bar_x0) * step / n_steps
    draw.polygon(
        [(x, bar_y - 8), (x - 7, bar_y - 22), (x + 7, bar_y - 22)],
        fill=(255, 255, 255, 255),
    )

    # pulse border at the beginning of each phase (keyframe cue)
    local = step - phase["start"]
    if local < 6:
        alpha = int(255 * (1 - local / 6))
        draw.rectangle((2, 2, w - 3, h - 3), outline=(*color, alpha), width=10)

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


def phase_tip(name: str) -> str:
    """Phase tip with the actual key labels of this run filled in."""
    tip = PHASE_INFO.get(name, ("", name, ""))[2]
    try:
        return tip.format(**KEY_LABELS)
    except (KeyError, IndexError):
        return tip


def keymap_labels() -> dict[str, str]:
    """Read hw3/keymap.json (if present) and map action name -> key label."""
    path = Path("hw3/keymap.json")
    if not path.exists():
        return {}
    raw_map = {
        2490368: "↑", 2621440: "↓", 2424832: "←", 2555904: "→",
        13: "Enter", 27: "Esc", 32: "Space",
    }
    out: dict[str, str] = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for action, entry in data.items():
        raw = int(entry["raw"])
        asc = int(entry.get("ascii", 0))
        out[action] = raw_map.get(raw, chr(asc).upper() if 32 <= asc <= 126 else f"raw:{raw}")
    return out


def draw_intro(keys: list[dict], n_steps: int, w: int = 1280, h: int = 960) -> np.ndarray:
    img = Image.new("RGB", (w, h), (12, 14, 18))
    d = ImageDraw.Draw(img)
    f_title = find_font(58, bold=True)
    f_sub = find_font(30)
    f_lab = find_font(26)
    f_key = find_font(40, bold=True)
    f_note = find_font(25)

    d.text((80, 104), "单方块抓取 · 按键教学版", font=f_title, fill=(255, 255, 255))
    d.text(
        (80, 192),
        f"0.5× 慢放（原速 10 Hz）· 每一拍 = 按一个键 · 共 {n_steps} 拍",
        font=f_sub, fill=(200, 205, 215),
    )
    d.text((80, 240), "按键说明（与你的 hw3/keymap.json 一致）", font=f_lab, fill=(150, 200, 170))

    seen: dict[str, str] = {}
    for k in keys:
        if k["action"] != "noop" and k["key"] not in seen:
            seen[k["key"]] = ACTION_TEXT.get(k["action"], k["action"])

    y = 276
    for key, text in seen.items():
        rounded_rect(d, (80, y, 164, y + 50), 9, (30, 34, 42), outline=(255, 255, 255), width=2)
        kb = d.textbbox((0, 0), key, font=f_key)
        d.text(
            (80 + (84 - (kb[2] - kb[0])) / 2, y + (50 - (kb[3] - kb[1])) / 2 - kb[1]),
            key, font=f_key, fill=(255, 255, 255),
        )
        d.text((186, y + 10), text, font=f_lab, fill=(230, 220, 160))
        y += 62

    # right column: how to use the video
    x0 = 700
    d.text((x0, 276), "怎么用它复原动作", font=f_lab, fill=(150, 200, 170))
    tips = [
        "① 先看一遍原速，再 0.5× 慢放跟练",
        "② 每拍只按一个键（视频里的 · 不按）",
        "③ 10 拍 = 1 秒；整条约 26 秒",
        "④ 抓取段 05–07 放慢反复练",
        "⑤ 视频按键 = 你 keymap.json 里的键",
    ]
    ty = 318
    for tip in tips:
        d.text((x0, ty), tip, font=f_note, fill=(210, 215, 225))
        ty += 46

    d.text((80, h - 150), "· 表示这一拍不按键（等机械臂稳定）", font=f_note, fill=(140, 145, 155))
    d.text((80, h - 112), "重点看阶段 05–07：下降到 z≈0.04 → 分步闭合 → 抬 5 cm 验证方块跟手",
           font=f_note, fill=(190, 195, 205))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_width or not cur:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def save_keys_txt(run_dir: Path, meta: dict, phases: list[dict]) -> None:
    keys = meta["keys"]
    labels = [k["key"] for k in keys]
    actions = [k["action"] for k in keys]
    lines = [
        "# 参考轨迹按键序列（10 Hz；每行一拍，每拍按一个键；· = 不按键）",
        f"# 总拍数 {len(labels)}，场景 seed {meta.get('seed')}，成功率 {meta.get('success')}",
        "",
        "## 按阶段分组（连按序列）",
    ]
    for pi, ph in enumerate(phases):
        name = ph["name"]
        title = PHASE_INFO.get(name, ("", name, ""))[1]
        lines.append("")
        lines.append(f"### 阶段 {pi + 1:02d} {title}  (step {ph['start']}–{ph['end'] - 1})")
        runs: list[list] = []
        for i in range(ph["start"], min(ph["end"], len(labels))):
            if runs and runs[-1][0] == labels[i]:
                runs[-1][1] += 1
            else:
                runs.append([labels[i], 1, actions[i]])
        lines.append("  " + "  ".join(f"{k}×{n}" for k, n, _ in runs))
        for k, n, a in runs:
            lines.append(f"    {k} × {n:2d}   {ACTION_TEXT.get(a, a)}")

    txt_path = run_dir / "expert_demo_keys.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = run_dir / "expert_demo_keys.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        f.write("step,key,action\n")
        for i, (k, a) in enumerate(zip(labels, actions)):
            f.write(f"{i},{k},{a}\n")
    print(f"  keys    : {txt_path.name} / {csv_path.name}")


def save_keyframe_sheet(phases: list[dict], n_steps: int, speed: float, path: Path,
                        key_counts: list | None = None) -> None:
    rows = len(phases)
    row_h = 84
    top = 210
    width, height = 1500, top + row_h * rows + 40
    img = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(img)
    f_title = find_font(46, bold=True)
    f_head = find_font(28, bold=True)
    f_cell = find_font(26)
    f_small = find_font(23)
    f_tiny = find_font(19)

    draw.text((40, 30), "单方块抓取参考轨迹 · 关键帧对照表", font=f_title, fill=(255, 255, 255))
    slow_s = n_steps / 10 / speed + (2 if key_counts is not None else 0)
    draw.text(
        (40, 92),
        f"0.5× 慢放视频：expert_demo_slow_annotated.mp4   ·   {n_steps} 个控制步 @10 Hz"
        f"（原速 {n_steps / 10:.1f}s，慢放后 {slow_s:.0f}s）",
        font=f_small,
        fill=(180, 190, 200),
    )
    draw.text(
        (40, 126),
        "提示：先在 top 视角对齐 XY，再只动 Z；夹爪分多步小按；绕过障碍走 x ≈ −0.28 走廊",
        font=f_small,
        fill=(150, 200, 170),
    )

    cols = [("编号", 40), ("阶段", 180), ("步范围", 640), ("要点 / 怎么按", 880)]
    draw.rectangle((0, top - 50, width, top - 4), fill=(40, 44, 52))
    for name, x0 in cols:
        draw.text((x0, top - 44), name, font=f_head, fill=(255, 255, 255))

    for i, ph in enumerate(phases):
        y = top + i * row_h
        bg = (26, 30, 36) if i % 2 == 0 else (32, 36, 44)
        draw.rectangle((0, y, width, y + row_h), fill=bg)
        c = PHASE_COLORS[i % len(PHASE_COLORS)]
        draw.rectangle((0, y, 8, y + row_h), fill=c)
        num, title, _ = PHASE_INFO.get(ph["name"], ("", ph["name"], ""))
        tip = phase_tip(ph["name"])
        draw.text((40, y + 26), num, font=f_cell, fill=(*c, 255))
        draw.text((180, y + 24), title, font=f_cell, fill=(255, 255, 255))
        if key_counts is not None:
            summary = "  ".join(f"{k}×{v}" for k, v in key_counts[i].most_common(3))
            draw.text((180, y + 56), summary or "（只停顿）", font=f_tiny, fill=(150, 180, 200))
        draw.text((640, y + 24), f"{ph['start']} – {ph['end'] - 1}", font=f_cell,
                  fill=(200, 210, 220))
        for j, line in enumerate(wrap_text(tip, f_small, width - 900, draw)[:2]):
            draw.text((880, y + 16 + j * 32), line, font=f_small, fill=(230, 220, 160))

    img.save(path)
    print(f"  keyframes: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotated 0.5x slow-motion demo video.")
    parser.add_argument("run_dir", nargs="?", type=Path, default=None,
                        help="Run directory (default: latest under datasets/expert_ref/single_cube).")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed (default 0.5).")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        candidates = []
        for base in (Path("datasets/expert_ref/keyed"), Path("datasets/expert_ref/single_cube")):
            if base.exists():
                candidates.extend(p for p in base.iterdir() if p.is_dir())
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    run_dir = run_dir.resolve()
    print(f"run_dir: {run_dir}")

    meta, ee, cube, grip = load_run(run_dir)
    phases = meta["phases"]
    n_steps = int(meta["steps"])
    raw_video = run_dir / "expert_demo_raw.mp4"
    src_video = raw_video if raw_video.exists() else run_dir / "expert_demo.mp4"
    frames = read_frames(src_video)
    print(f"source video: {src_video.name}  frames={len(frames)} steps={n_steps} phases={len(phases)}")

    repeat = max(1, int(round(1.0 / args.speed)))
    fonts = (
        find_font(42, bold=True),
        find_font(27),
        find_font(24),
        find_font(20),
    )

    # step -> phase index
    step_phase = np.zeros(n_steps, dtype=int)
    for i, ph in enumerate(phases):
        step_phase[ph["start"]:ph["end"]] = i

    # per-step key info (only for key-driven runs)
    keys = meta.get("keys")
    key_infos: list[dict] | None = None
    key_counts: list | None = None
    if keys:
        labels = [k["key"] for k in keys]
        actions = [k["action"] for k in keys]
        key_infos = []
        for i in range(n_steps):
            k = labels[i] if i < len(labels) else "·"
            a = actions[i] if i < len(actions) else "noop"
            nxt = labels[i + 1] if i + 1 < len(labels) else "结束"
            stream = labels[max(0, i - 9): i + 1]
            key_infos.append({
                "key": k,
                "text": ACTION_TEXT.get(a, a),
                "next_key": nxt,
                "stream": stream,
                "total": len(labels),
            })
        key_counts = []
        for ph in phases:
            c = Counter(labels[ph["start"]:ph["end"]])
            c.pop("·", None)
            key_counts.append(c)

        # fill in this run's actual key labels (gripper / arrows / reset)
        km = keymap_labels()
        used: dict[str, str] = {}
        for k in keys:
            used.setdefault(k["action"], k["key"])
        KEY_LABELS.update({
            "open": used.get("gripper_open", km.get("gripper_open", "M")),
            "close": used.get("gripper_close", km.get("gripper_close", "L")),
            "up": used.get("move_up", km.get("move_up", "↑")),
            "reset": km.get("reset", "R"),
            "record": km.get("record", "Space"),
            "end": km.get("end_episode", "Enter"),
        })

    out_path = run_dir / "expert_demo_slow_annotated.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if keys:
        intro = draw_intro(keys, n_steps)
        for _ in range(20):  # 2 s title card
            vw.write(intro)
    for i, f in enumerate(frames):
        phase_idx = int(step_phase[min(i, n_steps - 1)])
        ann = draw_frame(
            f, i, n_steps, phase_idx, phases, ee, cube,
            float(grip[min(i, len(grip) - 1)]), args.speed, fonts, i,
            key_infos[i] if key_infos else None,
        )
        for _ in range(repeat):
            vw.write(ann)
    vw.release()
    n_out = len(frames) * repeat + (20 if keys else 0)
    print(f"  video   : {out_path}  ({n_out} frames, {n_out / 10:.1f}s)")

    if key_infos is not None:
        save_keys_txt(run_dir, meta, phases)
    save_keyframe_sheet(phases, n_steps, args.speed, run_dir / "expert_demo_keyframes.png",
                        key_counts if key_infos else None)


if __name__ == "__main__":
    main()
