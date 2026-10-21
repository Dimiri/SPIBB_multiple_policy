"""Compute every number quoted in the paper from results/*.csv (so text and data cannot drift apart)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, TAB = ROOT / "results", ROOT / "tables"


def res(name):
    """Raw results: results/<name>.csv, or the compressed results/<name>.csv.gz shipped with the repo."""
    p = RES / f"{name}.csv"
    return p if p.exists() else RES / f"{name}.csv.gz"


def res_exists(name):
    return (RES / f"{name}.csv").exists() or (RES / f"{name}.csv.gz").exists()


def pct(x, nd=1):
    x = abs(float(x))
    if x == 0:
        return "0\\%"
    if 100 * x < 10 ** (-nd):
        return f"$<${10 ** (-nd):g}\\%"
    return f"{100 * x:.{nd}f}\\%"


def norm(d):
    d = d.copy()
    d["norm"] = (d.value - d.ref_value) / (d.opt_value - d.ref_value)
    d["worse"] = d.value < d.ref_value - 1e-9
    d["loss"] = np.minimum(d.value - d.ref_value, 0) / d.ref_value
    return d


def main():
    N = {}
    e0 = norm(pd.read_csv(res("e0")))
    v = e0.groupby(["method", "nwedge", "n_other"]).worse.mean()
    N["E0_NAIVE_MAX"] = pct(v.loc["mb_spibb"].max())
    N["E0_MB_MAX"] = pct(v.loc["mb_spibb_pg"].loc[[10, 20, 50]].max())

    for tag, f in [("E1", "e1"), ("E3", "e3")]:
        d = norm(pd.read_csv(res(f"{f}")))
        sp = d[d.method != "basic_rl"]
        viol = sp.groupby("method").worse.mean()
        worst = sp[sp.nwedge >= 10].groupby("method").loss.min()
        N[f"{tag}_MB_VIOL"] = pct(viol["mb_spibb_pg"], 2)
        N[f"{tag}_NAIVE_VIOL"] = pct(viol["mb_spibb"], 2)
        N[f"{tag}_OWN_VIOL"] = pct(viol["spibb_own"], 2)
        N[f"{tag}_P_VIOL"] = pct(viol["mb_spibb_p"], 2)
        N[f"{tag}_MB_WORST"] = pct(-worst["mb_spibb_pg"])
        N[f"{tag}_NAIVE_WORST"] = pct(-worst["mb_spibb"])
        if tag == "E3":
            p = sp[sp.method == "mb_spibb_p"]
            N["BETA_FAIL"] = f"{p[p.worse].beta_ref.median():.2f}"
            N["BETA_ALL"] = f"{p.beta_ref.median():.2f}"

    g = norm(pd.read_csv(res("e1")))
    g = g[g.nwedge == 20]
    m = g.groupby(["method", "ref", "n_total"]).norm.mean()
    f3 = lambda x: f"{x:.2f}"
    N["G150_MB_MED"], N["G150_MB_POOR"] = f3(m["mb_spibb_pg", "medium", 150]), f3(m["mb_spibb_pg", "poor", 150])
    N["G150_OWN_MED"], N["G150_OWN_POOR"] = f3(m["spibb_own", "medium", 150]), f3(m["spibb_own", "poor", 150])
    N["G150_FULL_MED"], N["G150_FULL_POOR"] = f3(m["spibb_own_full", "medium", 150]), f3(m["spibb_own_full", "poor", 150])
    N["G3000_OWN_GOOD"] = f3(m["spibb_own", "good", 3000])
    N["G3000_FULL_GOOD"] = f3(m["spibb_own_full", "good", 3000])
    N["G3000_MB_GOOD"] = f3(m["mb_spibb_pg", "good", 3000])

    r = norm(pd.read_csv(res("e3")))
    r = r[r.nwedge == 20]
    mr = r.groupby(["method", "n_total"]).norm.mean()
    N["R600_MB"], N["R600_OWN"], N["R600_FULL"] = (f3(mr["mb_spibb_pg", 600]), f3(mr["spibb_own", 600]),
                                                   f3(mr["spibb_own_full", 600]))

    # beta table (gridworld, medium)
    b = pd.read_csv(res("e1"))
    b = b[(b.ref == "medium") & (b.nwedge == 20)]
    rows = []
    for n in [30, 60, 150, 300, 600]:
        o = b[(b.n_total == n) & (b.method == "spibb_own")]
        po = b[(b.n_total == n) & (b.method == "mb_spibb")]
        mb = b[(b.n_total == n) & (b.method == "mb_spibb_pg")]
        rows.append(f"{n} & {o.frac_B.mean():.2f} & {po.frac_B.mean():.2f} & {o.beta_ref.mean():.3f} & "
                    f"{po.beta_ref.mean():.3f} & {po.beta_new.mean():.3f} & {mb.beta_new.mean():.3f} \\\\")
    N["BETA_TABLE"] = ("\\begin{tabular}{rrrrrrr}\\toprule\n"
                       " & \\multicolumn{2}{c}{$|\\B|/|S||A|$} & \\multicolumn{2}{c}{$\\beta_{\\B}(\\pref)$} & "
                       "\\multicolumn{2}{c}{$\\beta_{\\B}(\\pnew)$}\\\\\n"
                       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                       "$|\\D|$ & own & pooled & own & pooled & naive & MB \\\\\\midrule\n"
                       + "\n".join(rows) + "\n\\bottomrule\\end{tabular}")
    # pooled branch vs fallback of MB-SPIBB (identified by equality with SPIBB on the reference data)
    for tag, f in [("G", "e1"), ("R", "e3")]:
        d = pd.read_csv(res(f"{f}")); d = d[d.method.isin(["mb_spibb_pg", "spibb_own"])]
        w = d.pivot_table(index=["run", "n_total", "ref", "nwedge"], columns="method", values="value")
        rv = d.groupby(["run", "n_total", "ref", "nwedge"]).ref_value.first()
        fb = np.isclose(w.mb_spibb_pg, w.spibb_own)
        loss = (w.mb_spibb_pg - rv) / rv
        sel = (~fb) & (w.index.get_level_values("nwedge") >= 10)
        N[f"{tag}POOL_N"] = f"{int(sel.sum()):,}".replace(",", "{,}")
        N[f"{tag}POOL_VIOL"] = pct((loss[sel] < -1e-12).mean(), 2)
        N[f"{tag}POOL_WORST"] = pct(min(0, loss[sel].min()))
    # reference choice
    rc = pd.read_csv(TAB / "e3_refchoice_nw20.csv", index_col=0)
    acc = pd.read_csv(TAB / "e3_selection_accuracy.csv", index_col=0).iloc[:, 0]
    gacc = pd.read_csv(TAB / "e1_selection_accuracy.csv", index_col=0).iloc[:, 0]
    N["SEL9"], N["SEL150"], N["GSEL9"] = pct(acc[9], 0), pct(acc[150], 0), pct(gacc[9], 0)
    cols = list(rc.columns)
    N["SELV9"] = f3(rc.loc[9, "selected (LCB)"])
    N["P1V9"], N["P3V9"] = f3(rc.loc[9, cols[0]]), f3(rc.loc[9, cols[2]])
    N["P1V300"], N["P3V300"] = f3(rc.loc[300, cols[0]]), f3(rc.loc[300, cols[2]])
    grc = pd.read_csv(TAB / "e1_refchoice_nw20.csv", index_col=0)
    N["BCG"] = f3(grc.loc[9, "BC policy itself"])
    e1 = pd.read_csv(res("e1"), usecols=["ref", "ref_value", "opt_value"]).drop_duplicates()
    N["AVG_G"] = f3((e1.ref_value / e1.opt_value).mean())
    t = pd.read_csv(TAB / "e4_tau.csv").set_index(["tau", "n_total"])
    N["T1_60"] = pct(t.loc[(1.0, 60), "p_worse"])
    N["T05_150"] = pct(t.loc[(0.05, 150), "p_worse"])
    N["T2_150M"] = f3(t.loc[(0.2, 150), "mean"])
    N["T2_150V"] = pct(t.loc[(0.2, 150), "p_worse"])
    (TAB / "paper_numbers.json").write_text(json.dumps(N, indent=1))
    return N


if __name__ == "__main__":
    for k, v in main().items():
        print(k, "=", v if k != "BETA_TABLE" else "\n" + v)
