from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ordinal blue ramp, darker = more trusted (checked with the dataviz palette validator)
TIER_COLORS = {1: "#104281", 2: "#256abf", 3: "#5598e7", 4: "#86b6ef"}
TIER_NAMES = {1: "Tier 1  official / academic", 2: "Tier 2  reputable media", 3: "Tier 3  general website", 4: "Tier 4  forum / social"}
LABEL_ORDER = ["VERIFIED", "LIKELY", "CONTESTED", "UNVERIFIED"]
BAR_COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"


def bar_chart(names, values, colors, title):
    fig, ax = plt.subplots(figsize=(4.6, 2.3), dpi=200)
    positions = range(len(names))
    ax.barh(positions, values, color=colors, height=0.72)
    ax.set_yticks(list(positions), names, fontsize=8, color=INK)
    ax.invert_yaxis()
    biggest = max(values) if values and max(values) > 0 else 1
    for y, v in zip(positions, values):
        ax.text(v + biggest * 0.02, y, str(v), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, biggest * 1.15)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=6)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def tier_chart(sources):
    counts = [sum(1 for s in sources if s.tier == t) for t in (1, 2, 3, 4)]
    return bar_chart([TIER_NAMES[t] for t in (1, 2, 3, 4)], counts, [TIER_COLORS[t] for t in (1, 2, 3, 4)],
                     "Sources read, by trust tier")


def confidence_chart(groups):
    counts = [sum(1 for g in groups if g.confidence == label) for label in LABEL_ORDER]
    return bar_chart([label.title() for label in LABEL_ORDER], counts, BAR_COLOR, "Claim groups, by confidence label")
