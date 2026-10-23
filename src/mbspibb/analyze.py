"""Figures and tables for the paper.  python -m mbspibb.analyze  (reads results/*.csv, writes figures/ and tables/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "tables"


def res(name):
    """Raw results: results/<name>.csv, or the compressed results/<name>.csv.gz shipped with the repo."""
    p = RES / f"{name}.csv"
    return p if p.exists() else RES / f"{name}.csv.gz"


def res_exists(name):
    return (RES / f"{name}.csv").exists() or (RES / f"{name}.csv.gz").exists()
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)

# fixed method -> style mapping (colour follows the entity; marker/dash as secondary encoding)
STYLE = {
    "mb_spibb_pg":       dict(label="MB-SPIBB (ours)",            color="#2a78d6", marker="o", ls="-"),
    "mb_spibb_pg_select": dict(label="MB-SPIBB + selection (ours)", color="#1baf7a", marker="D", ls="-"),
    "mb_spibb_p":        dict(label="pessimistic, no guard",       color="#008300", marker="h", ls="-."),
    "mb_spibb":          dict(label="Pooled SPIBB (naive)",        color="#eb6834", marker="s", ls="--"),
    "spibb_own":         dict(label="SPIBB, reference data only",  color="#6250d6", marker="^", ls="-."),
    "spibb_own_full":    dict(label="SPIBB, all data from ref.",   color="#e87ba4", marker="v", ls=":"),
    "spibb_bc":          dict(label="SPIBB, BC-mixture ref.",      color="#eda100", marker="P", ls="--"),
    "basic_rl":          dict(label="Basic RL (pooled)",           color="#e34948", marker="x", ls=":"),
}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "pdf.fonttype": 42, "font.family": "serif",
})


def cvar(x, q):
    x = np.sort(np.asarray(x)); k = max(1, int(np.ceil(q * len(x))))
    return x[:k].mean()


def summarize(d, by):
    g = d.groupby(by)
    return g.apply(lambda t: pd.Series(dict(
        mean=t.norm.mean(), sem=t.norm.std(ddof=1) / np.sqrt(len(t)), cvar1=cvar(t.norm, 0.01),
        cvar10=cvar(t.norm, 0.10), p_worse=(t.value < t.ref_value - 1e-9).mean(), n=len(t))), include_groups=False
    ).reset_index()


def line(ax, s, x, y, methods, xlog=True):
    for m in methods:
        t = s[s.method == m].sort_values(x)
        if t.empty:
            continue
        st = STYLE[m]
        ax.plot(t[x], t[y], color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
    if xlog:
        ax.set_xscale("log")


def add_ref(ax, y=0.0):
    ax.axhline(y, color=INK, lw=0.8, zorder=0)


# --------------------------------------------------------------------------- E0 counterexample
def fig_e0():
    d = pd.read_csv(res("e0"))
    d["norm"] = d.value - d.ref_value
    s = summarize(d, ["method", "nwedge", "n_other"])
    ms = ["mb_spibb", "mb_spibb_pg", "spibb_own", "basic_rl"]
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.55), constrained_layout=True)
    line(axs[0], s[(s.nwedge == 10) | s.nwedge.isna()], "n_other", "p_worse", ms)
    axs[0].set(xlabel="foreign trajectories", ylabel=r"$P(\rho(\pi_{new}) < \rho(\pi_{ref}))$",
               title=r"(a) $N_\wedge=10$")
    t = s[s.n_other == 80]
    for m in ms[:3]:
        u = t[t.method == m].sort_values("nwedge"); st = STYLE[m]
        axs[1].plot(u.nwedge, u.p_worse, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
    axs[1].set_xscale("log"); axs[1].set_xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
    axs[1].minorticks_off(); axs[1].set(xlabel=r"$N_\wedge$", title="(b) 80 foreign trajectories")
    for ax in axs:
        ax.set_ylim(-0.01, 0.25)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(FIG / "e0_counterexample.pdf")
    s.to_csv(TAB / "e0_summary.csv", index=False)
    plt.close(fig)


# --------------------------------------------------------------------------- E1 / E3 data size
def normalise(d):
    gap = (d.opt_value - d.ref_value).replace(0, np.nan)
    d["norm"] = (d.value - d.ref_value) / gap
    return d


def fig_size(exp, refs, titles, nw=20, fname=None):
    d = normalise(pd.read_csv(res(f"{exp}")))
    d = d[(d.nwedge == nw) | d.nwedge.isna()]
    s = summarize(d, ["ref", "method", "n_total"])
    s.to_csv(TAB / f"{exp}_summary_nw{nw}.csv", index=False)
    ms = ["mb_spibb_pg", "mb_spibb", "spibb_own", "spibb_own_full", "basic_rl"]
    fig, axs = plt.subplots(2, len(refs), figsize=(7.0, 2.9), constrained_layout=True, sharex=True)
    for c, (ref, ttl) in enumerate(zip(refs, titles)):
        t = s[s.ref == ref]
        line(axs[0, c], t, "n_total", "mean", ms); add_ref(axs[0, c])
        line(axs[1, c], t, "n_total", "cvar1", ms); add_ref(axs[1, c])
        axs[0, c].set_title(ttl)
        axs[1, c].set_xlabel("trajectories in $D$ (all policies)")
        tp = t[t.method.isin([m for m in ms if m != "basic_rl"])]
        axs[1, c].set_ylim(max(-1.0, tp.cvar1.min() - 0.08), 1.05)
        axs[0, c].set_ylim(max(-1.0, min(-0.08, tp["mean"].min() - 0.08)), 1.05)
    axs[0, 0].set_ylabel("mean")
    axs[1, 0].set_ylabel("1%-CVaR")
    fig.supylabel("normalised performance", fontsize=7)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(FIG / (fname or f"{exp}_size.pdf"))
    plt.close(fig)
    return s


# --------------------------------------------------------------------------- reference choice
def fig_refchoice(exp, refs, labels, fname, nw=20):
    d = pd.read_csv(res(f"{exp}"))
    d = d[(d.nwedge == nw)]
    d["abs"] = d.value / d.opt_value
    rows = []
    for ref, lab in zip(refs, labels):
        t = d[(d.ref == ref) & (d.method == "mb_spibb_pg")]
        rows.append(t.groupby("n_total")["abs"].mean().rename(f"fixed: {lab}"))
    t = d[(d.ref == refs[0]) & (d.method == "spibb_bc")]
    rows.append(t.groupby("n_total")["abs"].mean().rename("BC mixture"))
    t = d[(d.ref == refs[0]) & (d.method == "mb_spibb_pg_select")]
    rows.append(t.groupby("n_total")["abs"].mean().rename("selected (LCB)"))
    best = d[d.ref == refs[0]].groupby("n_total").apply(lambda u: (u.best_value / u.opt_value).mean(),
                                                         include_groups=False).rename("best behavior policy")
    bc = d[d.ref == refs[0]].groupby("n_total").apply(lambda u: (u.bc_value / u.opt_value).mean(),
                                                       include_groups=False).rename("BC policy itself")
    tab = pd.concat(rows + [best, bc], axis=1)
    tab.to_csv(TAB / f"{exp}_refchoice_nw{nw}.csv")
    sel_acc = d[(d.ref == refs[0]) & (d.method == "mb_spibb_pg_select")]
    acc = pd.Series(np.isclose(sel_acc.selected_value, sel_acc.best_value), index=sel_acc.index).groupby(sel_acc.n_total).mean()
    acc.rename("selection accuracy").to_csv(TAB / f"{exp}_selection_accuracy.csv")

    fig, ax = plt.subplots(figsize=(3.4, 2.2), constrained_layout=True)
    cols = ["#6250d6", "#2a78d6", "#3fb6e6", "#eda100", "#1baf7a"]
    marks = ["^", "o", "s", "P", "D"]
    for col, c, mk in zip(tab.columns[:-2], cols, marks):
        big = col.startswith("selected")
        ax.plot(tab.index, tab[col], color=c, marker=mk, label=col, ms=2.5 if big else 3.5,
                lw=1.0 if big else 1.4, ls="-" if not col.startswith("fixed") or c != "#3fb6e6" else (0, (4, 2)))
    ax.plot(tab.index, tab["best behavior policy"], color=INK, ls="--", lw=0.9, label="best behavior policy")
    ax.plot(tab.index, tab["BC policy itself"], color=MUTED, ls=":", lw=0.9, label=r"$\rho(\hat\pi_{mix})$ (BC policy)")
    ax.set_xscale("log"); ax.set(xlabel="trajectories in $D$", ylabel=r"$\rho(\pi_{new})/\rho(\pi^*)$")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="outside lower center")
    fig.savefig(FIG / fname)
    plt.close(fig)
    return tab, acc


# --------------------------------------------------------------------------- E2 share of reference data
def fig_e2(n=300, nw=20):
    d = normalise(pd.read_csv(res("e2")))
    d = d[(d.n_total == n) & ((d.nwedge == nw) | d.nwedge.isna())]
    s = summarize(d, ["partner", "method", "share"])
    s.to_csv(TAB / f"e2_summary_n{n}_nw{nw}.csv", index=False)
    ms = ["mb_spibb_pg", "mb_spibb", "mb_spibb_p", "spibb_own", "basic_rl"]
    parts = [("good", "(a) rest from good policy"), ("poor", "(b) rest from poor policy"),
             ("both", "(c) rest from both")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.7), constrained_layout=True, sharey=True)
    for ax, (p, ttl) in zip(axs, parts):
        t = s[s.partner == p]
        line(ax, t, "share", "mean", ms, xlog=False); add_ref(ax)
        ax.set(title=ttl, xlabel=r"share of $D$ collected by $\pi_{ref}$")
    axs[0].set_ylabel("normalised perf. (mean)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside right center", frameon=False)
    fig.savefig(FIG / "e2_share.pdf")
    plt.close(fig)
    return s


# --------------------------------------------------------------------------- beta diagnostics
def table_beta(exp, ref, nw=20):
    d = pd.read_csv(res(f"{exp}"))
    d = d[(d.ref == ref) & (d.nwedge == nw) & d.method.isin(["spibb_own", "mb_spibb", "mb_spibb_p", "mb_spibb_pg"])]
    t = d.groupby(["n_total", "method"])[["beta_ref", "beta_new", "frac_B"]].mean().unstack("method")
    t.to_csv(TAB / f"{exp}_beta_{ref}_nw{nw}.csv")
    return t


def fig_e4():
    d = normalise(pd.read_csv(res("e4")))
    d["worse"] = d.value < d.ref_value - 1e-9
    d["loss"] = np.minimum(d.value - d.ref_value, 0) / d.ref_value
    s = d.groupby(["tau", "n_total"]).agg(mean=("norm", "mean"), cvar1=("norm", lambda x: cvar(x, 0.01)),
                                          p_worse=("worse", "mean"), worst=("loss", "min")).reset_index()
    s.to_csv(TAB / "e4_tau.csv", index=False)
    seq = ["#cfe0f7", "#9fc1ee", "#6aa0e3", "#2a78d6", "#1d5aa6", "#123b70"]  # one hue, light -> dark
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.6), constrained_layout=True)
    for c, (tau, t) in zip(seq, s.groupby("tau")):
        lab = r"$\tau$=" + ("1 (no guard)" if tau >= 1 else f"{tau:g}")
        axs[0].plot(t.n_total, t["mean"], color=c, marker="o", label=lab)
        axs[1].plot(t.n_total, t.p_worse, color=c, marker="o", label=lab)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("trajectories in $D$")
    axs[0].set_ylabel("mean norm. perf."); axs[1].set_ylabel(r"$P(\rho(\pi_{new})<\rho(\pi_{ref}))$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(FIG / "e4_tau.pdf"); plt.close(fig)
    return s


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 30)
    if (res("e0")).exists():
        fig_e0(); print(pd.read_csv(TAB / "e0_summary.csv").query("nwedge==10 or nwedge!=nwedge").round(3))
    if (res("e1")).exists():
        print(fig_size("e1", ["poor", "medium", "good"],
                       [r"(a) $\pi_{ref}$ = poor", r"(b) $\pi_{ref}$ = medium", r"(c) $\pi_{ref}$ = good"]).round(3))
        print(fig_refchoice("e1", ["poor", "medium", "good"], ["poor", "medium", "good"], "e1_refchoice.pdf"))
        print(table_beta("e1", "medium").round(4)); print(table_beta("e1", "poor").round(4))
    if (res("e2")).exists():
        print(fig_e2().round(3))
    if (res("e3")).exists():
        print(fig_size("e3", ["pi1", "pi2", "pi3"],
                       [r"(a) $\pi_{ref}=\pi_1$ ($\lambda$=0.3)", r"(b) $\pi_{ref}=\pi_2$ ($\lambda$=0.6)",
                        r"(c) $\pi_{ref}=\pi_3$ ($\lambda$=0.9)"]).round(3))
        print(fig_refchoice("e3", ["pi1", "pi2", "pi3"], [r"$\pi_1$", r"$\pi_2$", r"$\pi_3$"], "e3_refchoice.pdf"))
        print(table_beta("e3", "pi1").round(4))
    if (res("e4")).exists():
        print(fig_e4().round(4))
