"""ChurnIQ local analytics dashboard."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
    RocCurveDisplay, roc_auc_score)

from train_model import load_data

ROOT = Path(__file__).resolve().parent
MODEL_PATH, METRICS_PATH = ROOT / "models/churn_pipeline.joblib", ROOT / "models/metrics.json"
PRED_PATH = ROOT / "models/test_predictions.csv"
st.set_page_config(page_title="ChurnIQ | Retention Intelligence", page_icon="C", layout="wide", initial_sidebar_state="expanded")
plt.rcParams.update({
    "figure.figsize": (7.2, 3.5), "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E3EAF3", "axes.labelcolor": "#61718A", "text.color": "#172B4D",
    "xtick.color": "#61718A", "ytick.color": "#61718A", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#EAF0F6", "grid.linewidth": 0.8,
})
st.markdown("""
<style>
:root{--ink:#172B4D;--muted:#66758B;--blue:#2563EB;--line:#E3EAF3;--paper:#FFFFFF;--canvas:#F4F7FB}
.stApp{background:var(--canvas);color:var(--ink)}
.block-container{max-width:1380px;padding:1.5rem 2.25rem 3rem}
header[data-testid="stHeader"]{background:rgba(244,247,251,.88)}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.025em}
p,li{color:#33445D}
section[data-testid="stSidebar"]{background:#10243B;border-right:1px solid #1E3856}
section[data-testid="stSidebar"] *{color:#EAF1FA}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:10px;padding:.35rem .45rem}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#1B3655}
.side-brand{display:flex;align-items:center;gap:12px;padding:.35rem .25rem 1.5rem}
.side-mark{width:40px;height:40px;display:grid;place-items:center;border-radius:12px;background:linear-gradient(145deg,#4F8DFF,#2357C6);font-weight:800;color:white!important;box-shadow:0 7px 18px #06172d66}
.side-name{font-size:1.1rem;font-weight:750;line-height:1.15;color:#fff!important}
.side-caption{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:#93A8C2!important;margin-top:4px}
.topline{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin:0 0 2.1rem;padding-bottom:1rem;border-bottom:1px solid var(--line)}
.topline-brand{display:flex;align-items:center;gap:12px}
.main-mark{width:42px;height:42px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#4F8DFF,#2357C6);font-size:1rem;font-weight:800;color:#fff!important;box-shadow:0 6px 16px #2357c633}
.main-brand-name{font-size:1.38rem;font-weight:800;line-height:1.1;letter-spacing:-.035em;color:var(--ink)!important}
.topline-label{font-size:.69rem;font-weight:650;letter-spacing:.105em;color:#64748B;text-transform:uppercase;margin-top:4px}
.status-pill{border:1px solid #B8E4D1;background:#ECFDF5;color:#13734C!important;border-radius:99px;padding:.35rem .7rem;font-size:.76rem;font-weight:650;white-space:nowrap}
.page-kicker{color:var(--blue)!important;font-size:.72rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;margin:0 0 .45rem}
.page-title{color:var(--ink)!important;font-size:2.1rem;font-weight:760;letter-spacing:-.045em;line-height:1.13;margin:0 0 .55rem}
.page-description{color:var(--muted)!important;font-size:1rem;line-height:1.6;max-width:850px;margin:0 0 1.55rem}
[data-testid="stMetric"]{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:1.05rem 1.15rem;box-shadow:0 3px 12px #1c35520a;min-height:112px}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.82rem;font-weight:600}
[data-testid="stMetricValue"]{color:var(--ink)!important;font-size:1.85rem;font-weight:740}
[data-testid="stMetricDelta"]{font-size:.76rem}
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line);border-radius:16px}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#fff!important;border:1px solid #D8E2EE!important;border-radius:10px!important;color:#172B4D!important;min-height:42px}
[data-testid="stSelectbox"] [data-baseweb="select"] *{color:#172B4D!important}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:#64748B!important}
[data-testid="stNumberInput"] input{background:#fff!important;color:#172B4D!important;border-color:#D8E2EE!important;border-radius:9px!important}
[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#172B4D!important}
div[data-testid="stMarkdownContainer"] h3{margin-top:.3rem}
.section-note{color:var(--muted);font-size:.86rem;margin-top:-.2rem;margin-bottom:.8rem}
@media(max-width:800px){.block-container{padding:1.1rem 1rem 2rem}.page-title{font-size:1.7rem}.topline{margin-bottom:1.4rem}}
</style>
""", unsafe_allow_html=True)

def page_header(kicker, title, description):
    st.markdown(f"""<div class="page-kicker">{kicker}</div>
    <div class="page-title">{title}</div>
    <div class="page-description">{description}</div>""", unsafe_allow_html=True)

st.markdown("""<div class="topline"><div class="topline-brand"><div class="main-mark">C</div><div>
<div class="main-brand-name">ChurnIQ</div><div class="topline-label">Customer retention intelligence · IBM Telco case study</div></div></div>
<div class="status-pill">● &nbsp; Local analytics</div></div>""", unsafe_allow_html=True)

@st.cache_data
def dataset(): return load_data()

@st.cache_resource
def model_artifact(): return joblib.load(MODEL_PATH)

def metadata(): return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.markdown("""<div class="side-brand"><div class="side-mark">C</div><div>
    <div class="side-name">ChurnIQ</div><div class="side-caption">Retention intelligence</div></div></div>""", unsafe_allow_html=True)
    st.markdown("<div class='side-caption' style='padding:.2rem .45rem .6rem'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio("Section", ["Executive overview", "Explore data", "Model evaluation", "Predict a customer", "Business insights & limitations"])
    if page == "Predict a customer":
        st.divider()
        threshold = st.slider("Risk threshold", 0.10, 0.90, 0.50, 0.05,
                              help="Profiles at or above this probability are marked higher risk.")
    else:
        threshold = 0.50
    st.divider(); st.caption("Local educational case study · Historical sample data")

try:
    raw = dataset()
except Exception as exc:
    st.error(f"Dataset unavailable or invalid: {exc}")
    st.markdown("Place the real IBM CSV in `data/Telco-Customer-Churn.csv` (or the documented alternate filename).")
    st.code(".venv\\Scripts\\python.exe train_model.py\n.venv\\Scripts\\streamlit.exe run app.py")
    st.stop()
raw["ChurnLabel"] = raw.Churn.map({0: "Retained", 1: "Churned"})
num_cols = raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in raw.columns if c not in num_cols + ["ChurnLabel"]]

def bar_rate(data, col, ax):
    rates = data.groupby(col, dropna=False, observed=False).Churn.mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#3778E5", width=.64, zorder=3)
    ax.set_ylabel("Observed churn rate"); ax.set_xlabel(""); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", rotation=0); ax.grid(axis="y"); ax.grid(axis="x", visible=False)

if page == "Executive overview":
    page_header("01 / Executive overview", "A clearer view of customer retention", "Historical churn outcomes and contract patterns to help frame the retention problem. These are observed results, not individual risk predictions.")
    a,b,c,d=st.columns(4)
    a.metric("Total customers", f"{len(raw):,}"); b.metric("Observed churn rate", f"{raw.Churn.mean():.1%}")
    c.metric("Churned", f"{int(raw.Churn.sum()):,}"); d.metric("Retained", f"{int((raw.Churn==0).sum()):,}")
    left,right=st.columns(2)
    with left:
        st.markdown("#### Customer outcomes")
        fig,ax=plt.subplots(); counts=raw.ChurnLabel.value_counts().reindex(["Retained","Churned"]); ax.bar(counts.index,counts.values,color=["#A9BCD3","#3778E5"],width=.58,zorder=3); ax.set_ylabel("Customers"); ax.grid(axis="y"); ax.grid(axis="x",visible=False); st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("#### Observed churn by contract")
        if "Contract" in raw:
            fig,ax=plt.subplots(); bar_rate(raw,"Contract",ax); st.pyplot(fig); plt.close(fig)
        else: st.info("Contract field is not present.")
    st.info("Observed churn is a dataset outcome. Predicted risk is a model estimate for a profile. The two are different measures.")

elif page == "Explore data":
    page_header("02 / Explore", "Explore customer patterns", "Compare observed churn across tenure, monthly charges, and service or account categories. Associations in historical data do not establish causes.")
    l,r=st.columns(2)
    with l:
        st.markdown("#### Churn distribution")
        counts=raw.ChurnLabel.value_counts().reindex(["Retained","Churned"])
        fig,ax=plt.subplots(); ax.bar(counts.index,counts.values,color=["#A9BCD3","#3778E5"],width=.58,zorder=3); ax.set_ylabel("Customers"); ax.grid(axis="y"); ax.grid(axis="x",visible=False); st.pyplot(fig); plt.close(fig)
        if "tenure" in raw:
            st.markdown("#### Tenure by observed outcome")
            fig,ax=plt.subplots(); sns.boxplot(data=raw,x="ChurnLabel",y="tenure",order=["Retained","Churned"],ax=ax,color="#A9BCD3",width=.5,fliersize=2); ax.set_xlabel(""); ax.set_ylabel("Tenure (months)"); ax.grid(axis="x",visible=False); st.pyplot(fig); plt.close(fig)
    with r:
        st.markdown("#### Monthly charges by outcome")
        if "MonthlyCharges" in raw:
            fig,ax=plt.subplots(); sns.boxplot(data=raw,x="ChurnLabel",y="MonthlyCharges",order=["Retained","Churned"],ax=ax,color="#A9BCD3",width=.5,fliersize=2); ax.set_xlabel(""); ax.set_ylabel("Monthly charges"); ax.grid(axis="x",visible=False); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Churn across a categorical feature")
        choice=st.selectbox("Feature",cat_cols, index=(cat_cols.index("Contract") if "Contract" in cat_cols else 0))
        fig,ax=plt.subplots(); bar_rate(raw,choice,ax); st.pyplot(fig); plt.close(fig)
        grouped=raw.groupby(choice,dropna=False).Churn.agg(["mean","count"]).rename(columns={"mean":"Churn rate","count":"Customers"})
    st.dataframe(grouped.style.format({"Churn rate":"{:.1%}"}),width="stretch")
    st.caption(f"{len(raw):,} customers · {raw.drop(columns=['Churn','ChurnLabel']).shape[1]} input features · {int(raw.isna().sum().sum()):,} missing cells after numeric conversion. These are associations, not causal effects.")

elif page == "Model evaluation":
    page_header("03 / Model validation", "How well does the model separate churn risk?", "Compare both candidates on the same held-out test set. Model selection used a separate validation split; the final test results were not used to choose the winner.")
    if not METRICS_PATH.is_file() or not MODEL_PATH.is_file():
        st.warning("No trained model artifacts yet. Run `python train_model.py` after placing the CSV in data/."); st.stop()
    meta=metadata(); metrics=meta["metrics"]
    st.success(f"Final model selected by validation ROC-AUC: {meta['selected_model']}")
    st.caption(f"Threshold used for saved test classifications: {meta['threshold']:.2f}. Models were selected on a separate validation split; final metrics use the held-out test split. Evaluation run: {meta.get('evaluation_run_id', 'unknown')}.")
    table=pd.DataFrame({"ROC-AUC":[metrics[n]["roc_auc"] for n in metrics],"Precision":[metrics[n]["precision"] for n in metrics],"Recall":[metrics[n]["recall"] for n in metrics],"F1-score":[metrics[n]["f1"] for n in metrics],"Accuracy":[metrics[n]["accuracy"] for n in metrics]},index=metrics.keys())
    st.dataframe(table.style.format("{:.3f}"),width="stretch")
    selected=st.selectbox("Inspect saved candidate evaluation",list(metrics),index=list(metrics).index(meta["selected_model"]))
    m=metrics[selected]; cols=st.columns(5)
    for col,label,key in zip(cols,["ROC-AUC","Precision","Recall","F1-score","Accuracy"],["roc_auc","precision","recall","f1","accuracy"]): col.metric(label,f"{m[key]:.3f}")
    fig,ax=plt.subplots(); sns.heatmap(m["confusion_matrix"],annot=True,fmt="d",cmap="Blues",cbar=False,xticklabels=["Stay","Churn"],yticklabels=["Stay","Churn"],ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)
    if PRED_PATH.is_file():
        pred=pd.read_csv(PRED_PATH)
        run_matches = ("evaluation_run_id" in pred and
                       pred.evaluation_run_id.nunique() == 1 and
                       pred.evaluation_run_id.iloc[0] == meta.get("evaluation_run_id") and
                       MODEL_PATH.is_file() and
                       model_artifact().evaluation_run_id == meta.get("evaluation_run_id"))
        if run_matches and pred.actual_churn.nunique()==2 and pred.churn_probability.between(0,1).all():
            y=pred.actual_churn; p=pred.churn_probability
            fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(y,p,ax=ax); st.pyplot(fig); plt.close(fig)
            fig,ax=plt.subplots(); PrecisionRecallDisplay.from_predictions(y,p,ax=ax); st.pyplot(fig); plt.close(fig)
            st.caption(f"ROC-AUC verified from saved selected-model predictions: {roc_auc_score(y,p):.3f}.")
        else: st.warning("Saved prediction artifacts do not match this evaluation run. Retrain to regenerate a consistent artifact set.")
    st.write("False positives are customers flagged who stayed, potentially using outreach capacity unnecessarily. False negatives are customers who churned but were not flagged. A lower cutoff usually catches more churners while also increasing false alarms; use costs and capacity to choose it.")
    st.markdown("#### Model interpretation")
    summary = meta.get("model_interpretation", {}).get(selected)
    if summary:
        top = pd.DataFrame(summary["features"])
        top = top.rename(columns={"feature":"Encoded feature", "value":"Coefficient" if summary["kind"] == "coefficient" else "Importance"})
        if summary["kind"] == "coefficient":
            st.caption("Logistic Regression coefficients are on transformed inputs; numeric features are standardized. Positive values shift estimated churn odds upward, holding other encoded inputs fixed.")
            st.dataframe(top.style.format({"Coefficient":"{:.3f}"}),width="stretch")
        else:
            st.caption("Random Forest impurity importance summarizes how the fitted trees used transformed features. It is not a causal effect.")
            st.dataframe(top.style.format({"Importance":"{:.4f}"}),width="stretch")
    with st.expander("Classification reports"):
        st.json(metrics[selected]["classification_report"])

elif page == "Predict a customer":
    page_header("04 / Customer scoring", "Estimate a customer’s churn risk", "Enter a hypothetical customer profile to get a probability from the saved model. This estimate is not a certainty or a guarantee of future behavior.")
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file(): st.warning("Train the model first to enable prediction."); st.stop()
    mdl=model_artifact(); meta=metadata(); feature_cols=meta["feature_columns"]
    sample=raw.drop(columns=["Churn","ChurnLabel"],errors="ignore")
    st.write("Adjust the profile, then submit to score it with the same preprocessing and model used during training.")
    values={}
    with st.form("customer-profile"):
        cols=st.columns(3)
        for i,col in enumerate(feature_cols):
            series=sample[col]
            with cols[i%3]:
                if pd.api.types.is_numeric_dtype(series):
                    val=float(series.median()) if series.notna().any() else 0.0
                    low=0.0 if col.lower() in ("tenure","monthlycharges","totalcharges") else None
                    values[col]=st.number_input(col,min_value=low,value=max(val,low or val),step=1.0 if col.lower()=="tenure" else 5.0)
                else:
                    options=sorted(series.dropna().astype(str).unique().tolist())+(["Unknown"] if "Unknown" not in set(series.dropna().astype(str)) else [])
                    values[col]=st.selectbox(col,options)
        submitted=st.form_submit_button("Estimate churn probability",type="primary")
    if submitted:
        row=pd.DataFrame([values],columns=feature_cols)
        probability=float(mdl.predict_proba(row)[0,1])
        if not np.isfinite(probability) or not 0<=probability<=1: st.error("Model returned an invalid probability.")
        else:
            st.metric("Estimated churn probability",f"{probability:.1%}"); st.progress(probability)
            if probability>=threshold:
                st.warning(f"Higher risk at the selected {threshold:.0%} threshold. Consider reviewing customer feedback, recent support issues, and relevant retention options.")
            else: st.success(f"Below the selected {threshold:.0%} threshold. Continue normal service monitoring and support.")
            st.caption("This is a probability estimate, not certainty. Retention offers are not guaranteed to prevent churn.")

else:
    page_header("05 / Business context", "Turn model output into careful decisions", "Use risk estimates to prioritize a human review. Weigh potential customer value, outreach cost, campaign capacity, and the limits of historical data.")
    st.markdown("""**What the estimate means.** The classifier learns patterns associated with historical churn labels. It ranks profiles by estimated likelihood under this dataset and does not explain individual intent.

**Using scores responsibly.** Combine risk with customer value, outreach cost, campaign capacity, and customer preferences. Test interventions with controlled measurement; a high score alone does not show that an offer will change the outcome.

**Observed patterns in this loaded dataset.** Segment churn rates and median tenure/charges below are descriptive summaries. The highest and lowest rates are calculated directly from the real rows currently loaded; they are not causal effects.

**Limitations.** This public sample may be outdated, incomplete, or unrepresentative. Historical labels and random train/test splits may hide time drift. Missingness, measurement choices, sampling, and unequal service experiences can introduce bias. Monitor performance and calibration over time, validate on current data, and review outcomes across relevant groups before operational use.""")
    if "Contract" in raw:
        rates=raw.groupby("Contract").Churn.agg(["mean","count"]).sort_values("mean",ascending=False)
        top,bottom=rates.iloc[0],rates.iloc[-1]
        st.write(f"In this data, **{rates.index[0]}** customers had the highest observed churn rate ({top['mean']:.1%}, n={int(top['count']):,}); **{rates.index[-1]}** had the lowest ({bottom['mean']:.1%}, n={int(bottom['count']):,}).")
    if "tenure" in raw:
        medians=raw.groupby("Churn").tenure.median()
        st.write(f"Median tenure was {medians.get(1, float('nan')):.0f} months for churned customers and {medians.get(0, float('nan')):.0f} months for retained customers in this sample.")
    if "MonthlyCharges" in raw:
        medians=raw.groupby("Churn").MonthlyCharges.median()
        st.write(f"Median monthly charges were {medians.get(1, float('nan')):.2f} among churned customers and {medians.get(0, float('nan')):.2f} among retained customers (source data units).")
    if METRICS_PATH.is_file():
        meta=metadata(); st.write(f"The saved run selected **{meta['selected_model']}** using validation ROC-AUC ({meta['validation_roc_auc'][meta['selected_model']]:.3f}). Its held-out test ROC-AUC was {meta['metrics'][meta['selected_model']]['roc_auc']:.3f}; the reported classification threshold is {meta['threshold']:.2f}.")
        selected_pipe=model_artifact()
        estimator=selected_pipe.named_steps["model"]
        names=selected_pipe.named_steps["preprocess"].get_feature_names_out()
        if hasattr(estimator,"coef_"):
            weights=estimator.coef_[0]; ix=np.argsort(np.abs(weights))[::-1][:5]
            patterns=[f"{str(names[i]).replace('cat__','').replace('num__','')} ({'positive' if weights[i]>0 else 'negative'} coefficient)" for i in ix]
            st.write("Largest absolute Logistic Regression coefficients in the saved selected pipeline: " + "; ".join(patterns) + ". These are conditional model associations, not causes.")
    else: st.info("No trained evaluation is available yet. Run the training script to see model selection results.")
