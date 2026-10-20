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
PIPES_PATH = ROOT / "models/candidate_pipelines.joblib"
PRED_PATH = ROOT / "models/test_predictions.csv"
st.set_page_config(page_title="ChurnIQ | Retention Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1450px}
[data-testid="stMetric"]{background:#f3f7fc;border:1px solid #e0e9f5;padding:16px;border-radius:12px}
h1,h2,h3{color:#15324f}
</style>""", unsafe_allow_html=True)
st.title("📊 ChurnIQ")
st.caption("CUSTOMER RETENTION INTELLIGENCE · IBM Telco Customer Churn case study")

@st.cache_data
def dataset(): return load_data()

@st.cache_resource
def model_artifact(): return joblib.load(MODEL_PATH)

@st.cache_resource
def candidate_artifacts(): return joblib.load(PIPES_PATH)

def metadata(): return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Navigate")
    page = st.radio("Section", ["Executive overview", "Explore data", "Model evaluation", "Predict a customer", "Business insights & limitations"])
    threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05,
                          help="A probability at or above this cutoff is classified as higher risk.")
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
    rates.plot(kind="bar", ax=ax, color="#3977b8")
    ax.set_ylabel("Observed churn rate"); ax.set_xlabel(""); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", rotation=25)

if page == "Executive overview":
    st.subheader("A clearer view of customer retention")
    st.write("Churn prediction estimates which customer profiles resemble historical customers who left. It can help prioritize review; it cannot establish why someone leaves or guarantee an intervention will work.")
    a,b,c,d=st.columns(4)
    a.metric("Total customers", f"{len(raw):,}"); b.metric("Observed churn rate", f"{raw.Churn.mean():.1%}")
    c.metric("Churned", f"{int(raw.Churn.sum()):,}"); d.metric("Retained", f"{int((raw.Churn==0).sum()):,}")
    left,right=st.columns(2)
    with left:
        st.markdown("#### Customer outcomes")
        fig,ax=plt.subplots(); counts=raw.ChurnLabel.value_counts().reindex(["Retained","Churned"]); ax.bar(counts.index,counts.values,color=["#9db8d3","#3977b8"]); ax.set_ylabel("Customers"); st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("#### Observed churn by contract")
        if "Contract" in raw:
            fig,ax=plt.subplots(); bar_rate(raw,"Contract",ax); st.pyplot(fig); plt.close(fig)
        else: st.info("Contract field is not present.")
    st.info("Observed churn is a dataset outcome. Predicted risk is a model estimate for a profile; the two should not be confused.")

elif page == "Explore data":
    st.subheader("Exploratory data analysis")
    l,r=st.columns(2)
    with l:
        st.markdown("#### Churn distribution")
        counts=raw.ChurnLabel.value_counts().reindex(["Retained","Churned"])
        fig,ax=plt.subplots(); ax.bar(counts.index,counts.values,color=["#9db8d3","#3977b8"]); ax.set_ylabel("Customers"); st.pyplot(fig); plt.close(fig)
        if "tenure" in raw:
            st.markdown("#### Tenure by observed outcome")
            fig,ax=plt.subplots(); sns.boxplot(data=raw,x="ChurnLabel",y="tenure",order=["Retained","Churned"],ax=ax,color="#9db8d3"); ax.set_xlabel(""); ax.set_ylabel("Tenure (months)"); st.pyplot(fig); plt.close(fig)
    with r:
        st.markdown("#### Monthly charges by outcome")
        if "MonthlyCharges" in raw:
            fig,ax=plt.subplots(); sns.boxplot(data=raw,x="ChurnLabel",y="MonthlyCharges",order=["Retained","Churned"],ax=ax,color="#9db8d3"); ax.set_xlabel(""); ax.set_ylabel("Monthly charges"); st.pyplot(fig); plt.close(fig)
        st.markdown("#### Churn across a categorical feature")
        choice=st.selectbox("Feature",cat_cols, index=(cat_cols.index("Contract") if "Contract" in cat_cols else 0))
        fig,ax=plt.subplots(); bar_rate(raw,choice,ax); st.pyplot(fig); plt.close(fig)
        grouped=raw.groupby(choice,dropna=False).Churn.agg(["mean","count"]).rename(columns={"mean":"Churn rate","count":"Customers"})
    st.dataframe(grouped.style.format({"Churn rate":"{:.1%}"}),width="stretch")
    st.caption(f"Loaded dataset: {len(raw):,} rows · {raw.drop(columns=['Churn','ChurnLabel']).shape[1]} input features · {int(raw.isna().sum().sum()):,} missing cells after numeric conversion. These are associations, not causal effects.")

elif page == "Model evaluation":
    st.subheader("Held-out model evaluation")
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
                       PIPES_PATH.is_file() and
                       candidate_artifacts()[meta["selected_model"]].evaluation_run_id == meta.get("evaluation_run_id"))
        if run_matches and pred.actual_churn.nunique()==2 and pred.churn_probability.between(0,1).all():
            y=pred.actual_churn; p=pred.churn_probability
            fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(y,p,ax=ax); st.pyplot(fig); plt.close(fig)
            fig,ax=plt.subplots(); PrecisionRecallDisplay.from_predictions(y,p,ax=ax); st.pyplot(fig); plt.close(fig)
            st.caption(f"ROC-AUC verified from saved selected-model predictions: {roc_auc_score(y,p):.3f}.")
        else: st.warning("Saved prediction artifacts do not match this evaluation run. Retrain to regenerate a consistent artifact set.")
    st.write("False positives are customers flagged who stayed, potentially using outreach capacity unnecessarily. False negatives are customers who churned but were not flagged. A lower cutoff usually catches more churners while also increasing false alarms; use costs and capacity to choose it.")
    st.markdown("#### Model interpretation")
    if PIPES_PATH.is_file():
        pipe = candidate_artifacts()[selected]
        estimator = pipe.named_steps["model"]
        feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
        if hasattr(estimator, "coef_"):
            weights = estimator.coef_[0]
            top = pd.DataFrame({"Feature": feature_names, "Coefficient": weights})
            top["Direction"] = np.where(top.Coefficient >= 0, "Higher estimated churn", "Lower estimated churn")
            top["Magnitude"] = top.Coefficient.abs()
            st.caption("Logistic Regression coefficients are on transformed inputs; numeric features are standardized. Positive values shift estimated churn odds upward, holding other encoded inputs fixed.")
            st.dataframe(top.sort_values("Magnitude", ascending=False).drop(columns="Magnitude").head(15).style.format({"Coefficient":"{:.3f}"}),width="stretch")
        elif hasattr(estimator, "feature_importances_"):
            top = pd.DataFrame({"Feature": feature_names, "Importance": estimator.feature_importances_}).sort_values("Importance",ascending=False).head(15)
            st.caption("Random Forest impurity importance summarizes how the fitted trees used transformed features. It is not a causal effect.")
            st.dataframe(top.style.format({"Importance":"{:.4f}"}),width="stretch")
    with st.expander("Classification reports"):
        st.json(metrics[selected]["classification_report"])

elif page == "Predict a customer":
    st.subheader("Estimate an individual profile's churn risk")
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file(): st.warning("Train the model first to enable prediction."); st.stop()
    mdl=model_artifact(); meta=metadata(); feature_cols=meta["feature_columns"]
    sample=raw.drop(columns=["Churn","ChurnLabel"],errors="ignore")
    st.write("Choose customer attributes below. Each change submits a fresh estimate through the saved training pipeline.")
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
    st.subheader("Business insights and limitations")
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
        selected_pipe=candidate_artifacts()[meta["selected_model"]]
        estimator=selected_pipe.named_steps["model"]
        names=selected_pipe.named_steps["preprocess"].get_feature_names_out()
        if hasattr(estimator,"coef_"):
            weights=estimator.coef_[0]; ix=np.argsort(np.abs(weights))[::-1][:5]
            patterns=[f"{str(names[i]).replace('cat__','').replace('num__','')} ({'positive' if weights[i]>0 else 'negative'} coefficient)" for i in ix]
            st.write("Largest absolute Logistic Regression coefficients in the saved selected pipeline: " + "; ".join(patterns) + ". These are conditional model associations, not causes.")
    else: st.info("No trained evaluation is available yet. Run the training script to see model selection results.")
