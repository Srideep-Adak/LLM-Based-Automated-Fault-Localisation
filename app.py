"""Streamlit app.   Run:  streamlit run app.py"""
import json, os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fault Localizer", layout="wide")
st.title("Java Fault Localizer")
tab1, tab2 = st.tabs(["1. Evaluate result file (fl_*.json)", "2. Rank lines from a coverage file"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.write("Upload one or more result files such as `fl_gson.json` to see how often the real "
             "fault was in the top 1 / 3 / 5 / 10 lines.")
    files = st.file_uploader("Result JSON file(s)", type="json", accept_multiple_files=True, key="res")
    rows = []
    for f in files or []:
        try:
            data = json.load(f)
        except Exception as e:
            st.error(f"{f.name}: not valid JSON ({e})"); continue
        if not isinstance(data, list):
            st.error(f"{f.name}: this looks like a coverage file. Use tab 2 for it."); continue
        for r in data:
            o = r.get("ochiai") or {}
            t = r.get("tarantula") or {}
            if "error" in r:
                status, rank = "error: " + r["error"][:50], None
            elif o.get("avg") is None:
                status, rank = "fault outside search space", None
            else:
                status, rank = "ok", o["avg"]
            rows.append({"bug": r.get("id"), "failing tests": r.get("num_failing"),
                         "passing tests": r.get("num_passing"),
                         "search space (lines)": r.get("search_space_lines"),
                         "Ochiai rank of fault": rank, "Tarantula rank of fault": t.get("avg"),
                         "status": status})
    if rows:
        df = pd.DataFrame(rows).drop_duplicates("bug", keep="last")
        ok = df[df.status == "ok"]
        st.subheader(f"{len(df)} bugs loaded, {len(ok)} usable")
        c = st.columns(5)
        for col, k in zip(c, (1, 3, 5, 10)):
            n = int((ok["Ochiai rank of fault"] <= k).sum())
            col.metric(f"Top-{k}", f"{100 * n / max(len(ok), 1):.1f}%", f"{n} bugs")
        c[4].metric("Mean first rank", f"{ok['Ochiai rank of fault'].mean():.1f}" if len(ok) else "-")
        chart = pd.DataFrame({"% of bugs": [100 * (ok["Ochiai rank of fault"] <= k).mean() for k in (1, 3, 5, 10)]},
                             index=["Top-1", "Top-3", "Top-5", "Top-10"])
        st.bar_chart(chart)
        st.caption("Ties use the average rank, so a line tied with others at the top does not count as Top-1.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download table (CSV)", df.to_csv(index=False), "evaluation.csv")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Upload a **coverage JSON** (per-test covered lines, e.g. `fl_cache/Gson-1_modified.json`). "
             "The model predicts the **actual faulty line** and a ranked list of suspects. "
             "Several JSON layouts are accepted (see `fl_input.py`).")
    if not os.path.exists("model.joblib"):
        st.warning("`model.joblib` not found. Run `python train_model.py` first.")
    else:
        from fl_predict import load_bundle, predict, actual_fault
        from fl_input import normalize
        bundle = load_bundle()
        up = st.file_uploader("Coverage JSON", type="json", key="cov")
        sample = None
        if os.path.isdir("fl_cache"):
            names = sorted(n for n in os.listdir("fl_cache") if n.endswith("_modified.json"))
            sample = st.selectbox("...or pick a bug from fl_cache", ["-"] + names)
        raw = None
        try:
            if up is not None:
                raw = json.load(up)
            elif sample and sample != "-":
                raw = json.load(open(os.path.join("fl_cache", sample), encoding="utf-8"))
        except Exception as e:
            st.error(f"Cannot read the file: {e}")
        if raw is not None:
            try:
                cache, _ = normalize(raw)
                table, bug, info = predict(cache, bundle, top=20)
                res = actual_fault(table, bug, info)
            except ValueError as e:
                st.error(str(e)); res = None
            if res is not None:
                pf = res["predicted_fault"]
                st.success(f"Predicted fault: `{pf['file']}` line **{pf['line']}**  "
                           f"(probability {100 * pf['probability']:.0f}%, confidence: {pf['confidence']})")
                if res["also_check_nearby"]:
                    st.info("Also check nearby lines: " +
                            ", ".join(str(x["line"]) for x in res["also_check_nearby"]))
                for n in res["notes"]:
                    st.caption("Note: " + n)
                st.subheader("Chance the real fault is within the model's top lines")
                c = st.columns(3)
                for col, k in zip(c, (1, 3, 5)):
                    col.metric(f"Top-{k}", f"{100 * bug[k]:.0f}%")
                st.caption(f"{info['failing_tests']} failing / {info['passing_tests']} passing tests, "
                           f"{info['lines_covered']} covered lines. Percentages are estimated from "
                           f"{bundle['n_bugs']} Defects4J bugs - treat them as rough guides.")
                st.subheader("Most suspicious lines")
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.bar_chart(table.set_index(table["file"].str.split("/").str[-1] + ":" + table["line"].astype(str))["P(line is faulty)"])
                st.download_button("Download prediction (JSON)", json.dumps(res, indent=2),
                                   "prediction.json", "application/json")
        with st.expander("Model accuracy on Defects4J (leave-one-project-out)"):
            st.dataframe(pd.DataFrame(bundle["stats"]).T)