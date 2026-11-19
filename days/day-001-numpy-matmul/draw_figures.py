"""Still pictures for the matrix mental model.

The arrows are the axes. A point is a number of steps along those arrows.
When a matrix moves the arrows, the point moves with them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).parent / "figures"

X_COLOR = "#2C6BED"
Y_COLOR = "#E07A2F"
POINT_COLOR = "#111111"
OLD_COLOR = "#B9B9B9"


def new_axes(xlim, ylim):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(int(xlim[0]), int(xlim[1]) + 1))
    ax.set_yticks(range(int(ylim[0]), int(ylim[1]) + 1))
    ax.axhline(0, color="#222222", lw=1, zorder=1)
    ax.axvline(0, color="#222222", lw=1, zorder=1)
    ax.grid(True, color="#EEEEEE", zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#444444")
    return fig, ax


def arrow(ax, start, end, color, lw=2.4, scale=16, zorder=3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def point(ax, xy, label, dx=8, dy=8):
    ax.scatter([xy[0]], [xy[1]], s=55, color=POINT_COLOR, zorder=5)
    ax.annotate(
        label,
        xy,
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=11,
        color=POINT_COLOR,
        zorder=6,
    )


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def draw_point():
    fig, ax = new_axes((-0.6, 3.6), (-0.6, 4.2))
    arrow(ax, (0, 0), (2, 0), X_COLOR)
    arrow(ax, (2, 0), (2, 3), Y_COLOR)
    ax.text(1.0, 0.22, "2 × [1, 0]", color=X_COLOR, ha="center", fontsize=11)
    ax.text(2.15, 1.5, "3 × [0, 1]", color=Y_COLOR, ha="left", fontsize=11)
    point(ax, (2, 3), "P (2, 3)", dx=8, dy=-4)
    ax.set_title("P is 2 steps along x, then 3 steps along y", fontsize=12, pad=10)
    save(fig, "point.png")


def panel(ax, x_tip, y_tip, p, title, old=None):
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 2.8)
    ax.set_ylim(-1.6, 2.4)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1, 2])
    ax.axhline(0, color="#222222", lw=1, zorder=1)
    ax.axvline(0, color="#222222", lw=1, zorder=1)
    ax.grid(True, color="#EEEEEE", zorder=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12, pad=8)
    if old:
        arrow(ax, (0, 0), old[0], OLD_COLOR, lw=1.6, scale=12, zorder=2)
        arrow(ax, (0, 0), old[1], OLD_COLOR, lw=1.6, scale=12, zorder=2)
    arrow(ax, (0, 0), x_tip, X_COLOR)
    arrow(ax, (0, 0), y_tip, Y_COLOR)
    points_right = x_tip[0] > abs(x_tip[1])
    ax.annotate(
        "x-axis",
        x_tip,
        textcoords="offset points",
        xytext=(6, -18) if points_right else (12, -2),
        color=X_COLOR,
        fontsize=11,
    )
    ax.annotate(
        "y-axis",
        y_tip,
        textcoords="offset points",
        xytext=(-58, 8) if y_tip[0] < 0 else (8, 4),
        color=Y_COLOR,
        fontsize=11,
    )
    point(ax, p, "P", dx=8 if points_right else -18, dy=10)


def draw_pair(filename, after_title, x_tip, y_tip, p):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3), dpi=160)
    panel(axes[0], (1, 0), (0, 1), (1, 0), "before")
    panel(axes[1], x_tip, y_tip, p, after_title, old=((1, 0), (0, 1)))
    fig.tight_layout()
    save(fig, filename)


def main():
    draw_point()
    draw_pair("stretch.png", "stretch x by 2", (2, 0), (0, 1), (2, 0))
    draw_pair("quarter_turn.png", "quarter turn left", (0, 1), (-1, 0), (0, 1))


if __name__ == "__main__":
    main()
