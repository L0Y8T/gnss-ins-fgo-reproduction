"""plot.py —— 零依赖 SVG 绘图模块

不引入 matplotlib，直接用标准库拼 SVG，保证整个仓库零依赖。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
          "#9467bd", "#8c564b", "#17becf", "#e377c2"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

class Canvas:
    """极简 SVG 画布。"""

    def __init__(self, w, h, bg="#ffffff"):
        self.w, self.h = w, h
        self.parts = []
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">')
        self.parts.append(f'<rect width="{w}" height="{h}" fill="{bg}"/>')

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="middle", color="#222",
             weight="normal", rotate=None, family="sans-serif"):
        tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
                 f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}"{tr}>{_esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color="#333", width=1.5, dash=None,
             opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{color}" stroke-width="{width}"{d} '
                 f'opacity="{opacity}"/>')

    def rect(self, x, y, w, h, fill="none", stroke="#333", width=1.5, opacity=1.0):
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
                 f'opacity="{opacity}"/>')

    def circle(self, cx, cy, r, fill="#1f77b4", stroke="none", width=1.0):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def polyline(self, pts, color="#1f77b4", width=2.2, dash=None, opacity=1.0):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<polyline points="{p}" fill="none" stroke="{color}" '
                 f'stroke-width="{width}"{d} opacity="{opacity}" '
                 f'stroke-linejoin="round"/>')

    def save(self, path):
        body = "\n".join(self.parts) + "\n</svg>\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return path

def line_chart(path, series, title, xlabel, ylabel,
               width=900, height=560,
               logx=False, logy=False,
               hlines=None, legend_loc="upper right",
               annotate=None, ylim=None):
    """画折线图。

    ylim=(lo, hi) 显式指定纵轴范围、跳过自动留白。
    在对数轴上跨十几个数量级时必需：自动留白会把下限压到 0 以下，
    再取 log10 就会抛 domain error。
    """

    import math

    ml, mr, mt, mb = 90, 250, 60, 80
    pw, ph = width - ml - mr, height - mt - mb

    xs = [v for s in series for v in s["x"]]
    ys = [v for s in series for v in s["y"]]
    if hlines:
        ys += [h["y"] for h in hlines]
    if not xs or not ys:
        raise ValueError("没有数据可画")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if logx:
        xmin, xmax = max(xmin, 1e-9), max(xmax, 1e-9)
    if logy:
        ymin, ymax = max(ymin, 1e-9), max(ymax, 1e-9)

    ymin_data, ymax_data = ymin, ymax
    if ylim is not None:
        ymin, ymax = float(ylim[0]), float(ylim[1])
    else:
        pad = (ymax - ymin) * 0.10 if ymax > ymin else (abs(ymax) * 0.1 + 1.0)
        ymin -= pad * 0.35
        ymax += pad
        if ymin < 0 <= ymin_data:
            ymin = 0.0
    if xmin == xmax:
        xmax = xmin + 1

    def tx(v):
        if logx:
            a, b, c = math.log10(xmin), math.log10(xmax), math.log10(v)
            return ml + (c - a) / (b - a) * pw
        return ml + (v - xmin) / (xmax - xmin) * pw

    def ty(v):
        if logy:
            a, b, c = math.log10(ymin), math.log10(ymax), math.log10(v)
            return mt + ph - (c - a) / (b - a) * ph
        return mt + ph - (v - ymin) / (ymax - ymin) * ph

    cv = Canvas(width, height)

    cv.rect(ml, mt, pw, ph, fill="#fcfcfc", stroke="#999", width=1.2)

    n_ticks = 6
    for i in range(n_ticks + 1):
        v = ymin + (ymax - ymin) * i / n_ticks
        y = ty(v)
        cv.line(ml, y, ml + pw, y, color="#e2e2e2", width=1.0)
        cv.text(ml - 10, y + 4, f"{v:.2f}" if (ymax - ymin) < 10 else f"{v:.1f}",
                size=12, anchor="end", color="#555")

    xticks = sorted(set(xs))
    if len(xticks) > 14:
        step = max(1, len(xticks) // 12)
        xticks = xticks[::step] + [max(xs)]
    for v in xticks:
        x = tx(v)
        cv.line(x, mt + ph, x, mt + ph + 5, color="#999", width=1.2)
        cv.text(x, mt + ph + 22, f"{v:g}", size=12, color="#555")

    cv.text(width / 2, 28, title, size=18, weight="bold")
    cv.text(ml + pw / 2, height - 26, xlabel, size=14)
    cv.text(20, mt + ph / 2, ylabel, size=14, rotate=-90)

    if hlines:
        for h in hlines:
            y = ty(h["y"])
            col = h.get("color", "#777777")
            cv.line(ml, y, ml + pw, y, color=col, width=1.8, dash="7,5")
            cv.text(ml + 8, y - 7, h["label"], size=12, anchor="start", color=col)

    for i, s in enumerate(series):
        col = s.get("color", COLORS[i % len(COLORS)])
        dash = s.get("dash")
        pts = [(tx(x), ty(y)) for x, y in zip(s["x"], s["y"])]
        cv.polyline(pts, color=col, width=2.4, dash=dash)

        show_markers = len(pts) <= 30
        if show_markers:
            for (px, py) in pts:
                cv.circle(px, py, 3.6, fill=col)

    if annotate:
        for a in annotate:
            x, y = tx(a["x"]), ty(a["y"])
            cv.circle(x, y, 5.5, fill="none", stroke="#c0392b", width=2.0)
            cv.text(x, y - 12, a["text"], size=12, color="#c0392b", weight="bold")

    lx = ml + pw + 18
    ly = mt + 10
    cv.rect(lx - 6, ly - 14, mr - 20, 26 + 26 * (len(series) + (len(hlines) if hlines else 0)),
            fill="#ffffff", stroke="#ccc", width=1.0)
    for i, s in enumerate(series):
        col = s.get("color", COLORS[i % len(COLORS)])
        cv.line(lx, ly + 24 * i, lx + 30, ly + 24 * i, color=col, width=2.6,
                dash=s.get("dash"))
        cv.circle(lx + 15, ly + 24 * i, 3.6, fill=col)
        cv.text(lx + 38, ly + 24 * i + 4, s["name"], size=12.5, anchor="start")
    if hlines:
        base = len(series) * 24
        for j, h in enumerate(hlines):
            col = h.get("color", "#777777")
            yy = ly + base + 10 + 24 * j
            cv.line(lx, yy, lx + 30, yy, color=col, width=2.2, dash="7,5")
            cv.text(lx + 38, yy + 4, h["label"], size=12.5, anchor="start", color=col)

    return cv.save(path)

def bar_chart(path, groups, series_names, title, xlabel, ylabel,
              width=900, height=520, hlines=None):
    """分组柱状图。"""
    ml, mr, mt, mb = 90, 250, 60, 80
    pw, ph = width - ml - mr, height - mt - mb

    labels = [g[0] for g in groups]
    nser = len(series_names)
    vals = [[float(g[1 + i]) for g in groups] for i in range(nser)]

    ymax = max(max(v) for v in vals)
    if hlines:
        ymax = max(ymax, max(h["y"] for h in hlines))
    ymax *= 1.15

    def ty(v):
        return mt + ph - (v / ymax) * ph

    cv = Canvas(width, height)
    cv.rect(ml, mt, pw, ph, fill="#fcfcfc", stroke="#999", width=1.2)

    for i in range(7):
        v = ymax * i / 6
        y = ty(v)
        cv.line(ml, y, ml + pw, y, color="#e8e8e8", width=1.0)
        cv.text(ml - 10, y + 4, f"{v:.1f}", size=12, anchor="end", color="#555")

    gw = pw / len(labels)
    bw = gw * 0.72 / nser
    for j, lab in enumerate(labels):
        x0 = ml + gw * j + gw * 0.14
        for i in range(nser):
            v = vals[i][j]
            x = x0 + bw * i
            y = ty(v)
            cv.rect(x, y, bw * 0.9, mt + ph - y, fill=COLORS[i % len(COLORS)],
                    stroke="#ffffff", width=1.0)
            cv.text(x + bw * 0.45, y - 6, f"{v:.2f}", size=10.5, color="#333")
        cv.text(ml + gw * (j + 0.5), mt + ph + 22, lab, size=12.5, color="#444")

    cv.text(width / 2, 28, title, size=18, weight="bold")
    cv.text(ml + pw / 2, height - 26, xlabel, size=14)
    cv.text(20, mt + ph / 2, ylabel, size=14, rotate=-90)

    if hlines:
        for h in hlines:
            y = ty(h["y"])
            col = h.get("color", "#777777")
            cv.line(ml, y, ml + pw, y, color=col, width=2.0, dash="7,5")
            cv.text(ml + pw - 6, y - 7, h["label"], size=12, anchor="end", color=col)

    lx, ly = ml + pw + 18, mt + 10
    cv.rect(lx - 6, ly - 14, mr - 20, 26 + 26 * nser, fill="#ffffff",
            stroke="#ccc", width=1.0)
    for i, nm in enumerate(series_names):
        cv.rect(lx, ly + 24 * i - 8, 22, 14, fill=COLORS[i % len(COLORS)],
                stroke="none")
        cv.text(lx + 30, ly + 24 * i + 4, nm, size=12.5, anchor="start")

    return cv.save(path)

if __name__ == "__main__":

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_plot.svg")
    line_chart(out,
               [{"name": "曲线 A", "x": [1, 2, 3, 4, 5], "y": [5, 3, 4, 2, 1]},
                {"name": "曲线 B", "x": [1, 2, 3, 4, 5], "y": [2, 2.5, 2.2, 1.8, 1.5]}],
               "折线图自检", "X 轴", "Y 轴",
               hlines=[{"y": 2.0, "label": "参考线"}])
    print("已写出", out)
    print("用浏览器打开即可查看。")

