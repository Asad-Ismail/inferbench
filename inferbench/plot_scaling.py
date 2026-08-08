"""Render scaling-law figures from scaling_probe.py CSV output.

Requires CSVs under docs/data/ (not shipped). Produce them with scaling_probe.py, e.g.:

    python3 inferbench/scaling_probe.py --provider myapi --sweep context --out docs/data/ctx_ext.csv
    python3 inferbench/scaling_probe.py --provider myapi --sweep cachehit --context 200000 --out docs/data/hit_200.csv

Then:

    python3 inferbench/plot_scaling.py

Figures:
  scaling_context.png  TTFT vs prefill tokens, with the quadratic fit and the
                       linear->quadratic crossover.
  scaling_cachehit.png TTFT vs (1-hit) at 200k and 400k context, affine fit with the
                       nonzero floor, contrasted with the naive proportional line.
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "docs", "data")
OUT = os.path.join(HERE, os.pardir, "docs", "figures")

# Okabe-Ito, colorblind-safe. Identity, in fixed order — not cycled.
BLUE, ORANGE, VERM, GREEN, GRAY, INK = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#9AA0A6", "#202124"


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing {path}\n"
            f"Generate CSVs with scaling_probe.py --out docs/data/<name>.csv "
            f"(see plot_scaling.py docstring)."
        )
    return list(csv.DictReader(open(path)))


def _style(ax):
    ax.grid(True, color="#E8EAED", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BDC1C6")
    ax.tick_params(colors=INK, labelsize=9)


def context_fig():
    rows = [r for r in _load("ctx_ext.csv") if int(r["total_tokens"]) < 800000]  # drop extreme outliers
    L = np.array([float(r["prompt_tokens"]) for r in rows])
    y = np.array([float(r["ttft_p50_s"]) for r in rows])
    c = np.polyfit(L, y, 2)                      # c[0]=c2, c[1]=c1, c[2]=c0
    r2 = 1 - np.sum((y - np.polyval(c, L))**2) / np.sum((y - y.mean())**2)
    xstar = c[1] / c[0]                          # linear==quadratic term

    fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=150)
    _style(ax)
    xs = np.linspace(L.min(), L.max(), 400)
    # Decomposition: total = const a + linear term (b·L) + attention term (c·L²). The curve is
    # convex everywhere (no linear regime); the two terms simply cross at L* = b/c.
    ax.plot(xs / 1e3, c[1] * xs, color=GREEN, lw=1.5, ls="--", zorder=1, label="linear term  b·L")
    ax.plot(xs / 1e3, c[0] * xs**2, color=VERM, lw=1.5, ls="--", zorder=1, label="quadratic term  c·L²")
    ax.plot(xs / 1e3, np.polyval(c, xs), color=ORANGE, lw=2.4, zorder=2,
            label=f"total fit  a+bL+cL²  (R²={r2:.5f})")
    ax.scatter(L / 1e3, y, s=42, color=BLUE, zorder=3, label="measured (0% cache)")
    ax.axvline(xstar / 1e3, color=GRAY, lw=1.4, ls=":", zorder=1)
    ax.annotate(f"quadratic term overtakes\nlinear term at L*≈{xstar/1e3:.0f}k",
                xy=(xstar / 1e3, c[0] * xstar**2), xytext=(0.55, 0.28),
                textcoords="axes fraction", fontsize=8.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    ax.set_xlabel("prefill tokens (FRESH / uncached, thousands)", fontsize=10, color=INK)
    ax.set_ylabel("TTFT (s)", fontsize=10, color=INK)
    ax.set_title("prefill TTFT vs context length (0% cache)",
                 fontsize=12, color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(OUT, "scaling_context.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, (c[2], c[1], c[0], r2, xstar)


def cachehit_fig():
    def series(files):
        rows = [r for f in files for r in _load(f)]
        x = np.array([1 - float(r["intended_hit"]) for r in rows])
        y = np.array([float(r["ttft_p50_s"]) for r in rows])
        o = np.argsort(x)
        return x[o], y[o]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    _style(ax)
    fits = {}
    for label, files, col in [("200k context", ["hit.csv", "hit_200.csv"], BLUE),
                              ("400k context", ["hit_400.csv"], VERM)]:
        x, y = series(files)
        b, a = np.polyfit(x, y, 1)               # y = a + b*x
        r2 = 1 - np.sum((y - (a + b * x))**2) / np.sum((y - y.mean())**2)
        fits[label] = (a, b, r2)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, a + b * xs, color=col, lw=2, zorder=2,
                label=f"{label}: TTFT = {a:.1f} + {b:.1f}·(1−hit)  (floor {a:.1f}s)")
        ax.scatter(x, y, s=42, color=col, zorder=3)
        ax.scatter([0], [a], s=44, facecolors="white", edgecolors=col, lw=1.6, zorder=4)  # floor
    ax.set_xlabel("uncached fraction  (1 − cache-hit)", fontsize=10, color=INK)
    ax.set_ylabel("TTFT (s)", fontsize=10, color=INK)
    ax.set_title("prefill TTFT vs cache-hit rate",
                 fontsize=12, color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(OUT, "scaling_cachehit.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p, fits


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        p1, ctx = context_fig()
        p2, hit = cachehit_fig()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"context: c0={ctx[0]:.4g} c1={ctx[1]:.4e} c2={ctx[2]:.4e} R2={ctx[3]:.7f} L*={ctx[4]/1e3:.0f}k -> {p1}")
    for k, (a, b, r2) in hit.items():
        print(f"cachehit {k}: floor={a:.2f}s slope={b:.2f} R2={r2:.4f}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
