#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS vs VoxCPM2 压测对比图表生成
数据来源: benchmark.txt (vLLM-Omni 部署, 单卡 NVIDIA L20)

运行方式 (隔离 uv 环境, 不污染系统 Python):
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python matplotlib numpy
    .venv/bin/python gen_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# 1. 字体: 优先项目内 Noto Sans SC, 其次 Windows / 系统路径, 兜底默认
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
FONT_CANDIDATES = [
    HERE / "assets" / "NotoSansSC-Regular.otf",
    Path("/mnt/c/Windows/Fonts/Noto Sans SC (TrueType).otf"),
    Path("/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]

CN_FONT = None
for cand in FONT_CANDIDATES:
    if cand.exists():
        try:
            fm.fontManager.addfont(str(cand))
            CN_FONT = fm.FontProperties(fname=str(cand))
            plt.rcParams["font.family"] = CN_FONT.get_name()
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# --------------------------------------------------------------------------- #
# 2. 全局样式
# --------------------------------------------------------------------------- #
C_VOX = "#E45756"   # VoxCPM2 - 红
C_QWEN = "#4C78A8"  # Qwen3-TTS - 蓝
M_VOX, M_QWEN = "o", "s"
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# --------------------------------------------------------------------------- #
# 3. 数据 (从 benchmark.txt 人工解析, 每组 200 请求)
#    字段: 成功率% / RTF(mean,P50,P90,P95,P99,max) / Latency(mean,P50,P90,P95,P99,max)
#          wall_time(s) / 吞吐(req/s) / 成功吞吐(req/s) / 音频均长(s) / 音频总长(s) / 音频字节
# --------------------------------------------------------------------------- #
CONC = [1, 2, 3, 4, 5, 6, 10, 20]

VOX = {
    "name": "VoxCPM2",
    "succ":  [97.0, 97.0, 99.5, 96.0, 96.0, 97.0, 94.0, 25.0],
    "rtf_m":  [0.145, 0.162, 0.195, 0.206, 0.233, 0.281, 0.421, 0.970],
    "rtf_p50":[0.134, 0.157, 0.188, 0.202, 0.231, 0.273, 0.417, 0.923],
    "rtf_p90":[0.161, 0.180, 0.226, 0.232, 0.260, 0.323, 0.483, 1.399],
    "rtf_p99":[0.590, 0.230, 0.278, 0.271, 0.309, 0.448, 0.551, 1.788],
    "lat_m":  [0.550, 0.619, 0.728, 0.792, 0.896, 1.036, 1.530, 2.269],
    "lat_p50":[0.499, 0.573, 0.694, 0.771, 0.842, 1.007, 1.495, 2.366],
    "lat_p90":[0.808, 0.816, 1.044, 1.058, 1.238, 1.441, 2.141, 2.893],
    "lat_p95":[0.944, 1.028, 1.249, 1.263, 1.484, 1.678, 2.506, 2.922],
    "lat_p99":[2.663, 1.893, 1.748, 1.814, 2.188, 2.012, 2.779, 2.944],
    "thr":    [1.60, 2.86, 4.01, 4.49, 5.04, 5.39, 5.81, 6.87],   # 总吞吐
    "thr_s":  [1.55, 2.77, 3.99, 4.31, 4.83, 5.23, 5.46, 1.72],   # 成功吞吐
    "audio_s":[752.16, 750.24, 758.56, 754.40, 751.36, 733.44, 692.80, 126.72],
    "bytes":  [72215896, 72031576, 72830516, 72430848, 72139008, 70418776, 66517072, 12167320],
}

QWEN = {
    "name": "Qwen3-TTS-0.6B",
    "succ":  [100.0, 99.5, 99.5, 100.0, 100.0, 100.0, 98.0, 94.0],
    "rtf_m":  [0.168, 0.198, 0.223, 0.245, 0.264, 0.287, 0.374, 0.678],
    "rtf_p50":[0.163, 0.191, 0.219, 0.236, 0.256, 0.281, 0.362, 0.664],
    "rtf_p90":[0.184, 0.226, 0.250, 0.286, 0.308, 0.327, 0.408, 0.868],
    "rtf_p99":[0.232, 0.255, 0.301, 0.344, 0.356, 0.362, 0.900, 1.193],
    "lat_m":  [0.618, 0.725, 0.833, 0.868, 0.966, 1.062, 1.389, 2.281],
    "lat_p50":[0.579, 0.696, 0.789, 0.835, 0.920, 1.013, 1.312, 2.322],
    "lat_p90":[0.932, 1.001, 1.232, 1.247, 1.373, 1.596, 2.066, 2.758],
    "lat_p95":[1.091, 1.069, 1.366, 1.505, 1.601, 1.785, 2.317, 2.926],
    "lat_p99":[1.345, 1.439, 1.825, 1.841, 1.929, 2.157, 2.729, 2.980],
    "thr":    [1.61, 2.70, 3.52, 4.55, 5.11, 5.49, 6.89, 8.19],
    "thr_s":  [1.61, 2.69, 3.50, 4.55, 5.11, 5.49, 6.75, 7.69],
    "audio_s":[742.32, 733.68, 749.04, 719.52, 740.00, 745.36, 745.20, 659.20],
    "bytes":  [35640160, 35225396, 35962676, 34545760, 35528800, 35786080, 35778224, 31649872],
}

x = np.arange(len(CONC))


def line_pair(values_key, title, ylabel, ref=None, fmt="{:.2f}"):
    """通用双折线图"""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, VOX[values_key], color=C_VOX, marker=M_VOX, lw=2, ms=7,
            label="VoxCPM2")
    ax.plot(x, QWEN[values_key], color=C_QWEN, marker=M_QWEN, lw=2, ms=7,
            label="Qwen3-TTS-0.6B")
    if ref is not None:
        ax.axhline(ref, color="#888", ls=":", lw=1.5)
        ax.text(x[-1], ref, f"  y={ref}", color="#555", fontsize=9, va="bottom", ha="right")
    # 数值标注
    for arr, c, dy in [(VOX[values_key], C_VOX, 0.02), (QWEN[values_key], C_QWEN, -0.02)]:
        for xi, yi in zip(x, arr):
            ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points",
                        xytext=(0, 10 if dy > 0 else -14), fontsize=7.5,
                        color=c, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(CONC)
    style_ax(ax, title, "并发数 (Concurrency)", ylabel)
    ax.legend(fontsize=10, loc="best")
    return fig


# ---- 图1: 平均 RTF ----
fig = line_pair("rtf_m", "平均 RTF 随并发变化 (越低越好)", "RTF (Real-Time Factor)",
                ref=1.0, fmt="{:.3f}")
save(fig, "rtf_curve.png")

# ---- 图2: 成功吞吐 ----
fig = line_pair("thr_s", "成功吞吐量随并发变化 (越高越好)", "吞吐量 (req/s)", fmt="{:.2f}")
save(fig, "throughput_curve.png")

# ---- 图3: 平均延迟 ----
fig = line_pair("lat_m", "平均端到端延迟随并发变化 (越低越好)", "延迟 (秒)", fmt="{:.2f}")
save(fig, "latency_curve.png")

# ---- 图4: 成功率 ----
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(x, VOX["succ"], color=C_VOX, marker=M_VOX, lw=2, ms=7, label="VoxCPM2")
ax.plot(x, QWEN["succ"], color=C_QWEN, marker=M_QWEN, lw=2, ms=7, label="Qwen3-TTS-0.6B")
ax.axhline(95, color="#888", ls=":", lw=1.5)
ax.text(0, 95.3, "95% 可用性基线", color="#555", fontsize=9)
for arr, c in [(VOX["succ"], C_VOX), (QWEN["succ"], C_QWEN)]:
    for xi, yi in zip(x, arr):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), fontsize=7.5, color=c, ha="center")
# 高亮 VoxCPM2 并发20 崩塌
ax.annotate("VoxCPM2 并发20 崩塌\n成功率仅 25%",
            xy=(7, 25), xytext=(5.0, 48),
            fontsize=9, color=C_VOX, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_VOX))
ax.set_xticks(x)
ax.set_xticklabels(CONC)
ax.set_ylim(0, 108)
style_ax(ax, "请求成功率随并发变化 (越高越好)", "并发数 (Concurrency)", "成功率 (%)")
ax.legend(fontsize=10, loc="center right")
save(fig, "success_rate_curve.png")

# ---- 图5: 各并发度平均 RTF 柱状对比 ----
width = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.2))
b1 = ax.bar(x - width / 2, VOX["rtf_m"], width, color=C_VOX, label="VoxCPM2")
b2 = ax.bar(x + width / 2, QWEN["rtf_m"], width, color=C_QWEN, label="Qwen3-TTS-0.6B")
ax.axhline(1.0, color="#888", ls=":", lw=1.5)
ax.text(x[-1], 1.0, "  实时门槛 RTF=1.0", color="#555", fontsize=9, va="bottom", ha="right")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.2f}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    fontsize=7.5, ha="center")
ax.set_xticks(x)
ax.set_xticklabels(CONC)
style_ax(ax, "各并发度平均 RTF 对比 (越低越好)", "并发数 (Concurrency)", "RTF")
ax.legend(fontsize=10)
save(fig, "rtf_bar_per_concurrency.png")

# ---- 图6: 延迟分位数对比 (P50/P90/P99) ----
sel = [0, 5, 6, 7]  # 并发 1, 6, 10, 20
labels = [f"并发{CONC[i]}" for i in sel]
percentiles = [("P50", "lat_p50"), ("P90", "lat_p90"), ("P99", "lat_p99")]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
for ax, (pname, key) in zip(axes, percentiles):
    xv = np.arange(len(sel))
    ax.bar(xv - width / 2, [VOX[key][i] for i in sel], width, color=C_VOX, label="VoxCPM2")
    ax.bar(xv + width / 2, [QWEN[key][i] for i in sel], width, color=C_QWEN, label="Qwen3-TTS-0.6B")
    ax.set_title(f"延迟 {pname}", fontsize=12, fontweight="bold")
    ax.set_xticks(xv)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("延迟 (秒)")
axes[0].legend(fontsize=9)
fig.suptitle("端到端延迟分位数对比 (越低越好)", fontsize=14, fontweight="bold", y=1.02)
save(fig, "latency_percentiles.png")

# ---- 图7: 输出音频数据率 (字节/秒) ----
vox_rate = [b / s for b, s in zip(VOX["bytes"], VOX["audio_s"])]
qwen_rate = [b / s for b, s in zip(QWEN["bytes"], QWEN["audio_s"])]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.bar(x - width / 2, vox_rate, width, color=C_VOX, label="VoxCPM2")
ax.bar(x + width / 2, qwen_rate, width, color=C_QWEN, label="Qwen3-TTS-0.6B")
ax.axhline(48000, color="#888", ls=":", lw=1.5)
ax.text(0, 48500, "≈ 24kHz/16bit = 48 KB/s", color="#555", fontsize=9)
ax.axhline(96000, color="#aaa", ls=":", lw=1.5)
ax.text(0, 96500, "≈ 48kHz/16bit = 96 KB/s", color="#555", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(CONC)
style_ax(ax, "输出音频数据率 (字节/秒, 反映采样率/位深)",
         "并发数 (Concurrency)", "数据率 (Bytes/s)")
ax.legend(fontsize=10)
save(fig, "audio_datarate.png")

print("\n全部图表生成完毕 ->", OUT)
