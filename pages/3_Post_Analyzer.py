from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from src.data import initialize_state, load_hard_lexicon, load_model, load_soft_lexicon
from src.text_utils import clean_text, lexicon_score, top_matching_phrases

st.set_page_config(page_title="Post Analyzer", layout="wide")
st.title("Post Analyzer")
st.caption("Analyze a draft post with model-based prediction when available, otherwise lexicon-only diagnostics.")

posts_df, profiles_df, filtered, filters, warnings, _ = initialize_state()
for w in warnings:
    st.warning(w)
if posts_df.empty:
    st.error("No usable data loaded.")
    st.stop()

tfidf_obj, tfidf_path = load_model("tfidf")
kmeans_obj, kmeans_path = load_model("kmeans")
svd_obj, svd_path = load_model("svd")

hard_lex, hard_path = load_hard_lexicon()
soft_lex, soft_path = load_soft_lexicon()

model_available = tfidf_obj is not None and kmeans_obj is not None
if model_available:
    st.caption(f"Using models -> tfidf: {tfidf_path} | kmeans: {kmeans_path} | svd: {svd_path or 'not used'}")
else:
    st.warning("Model files are missing; running lexicon-only analyzer fallback.")

text = st.text_area("Paste title + body", height=200)

if st.button("Analyze post"):
    if not text.strip():
        st.warning("Please paste text first.")
        st.stop()

    lex = lexicon_score(text, hard_lex, soft_lex)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hard-skill framing", f"{lex['hard_norm']:.3f}")
    c2.metric("Soft-skill framing", f"{lex['soft_norm']:.3f}")
    c3.metric("Desirability index", f"{lex['desirability_index']:.3f}")
    st.write(f"Inferred class: **{lex['desirability_class']}**")

    st.subheader("Diagnostic evidence")
    st.write("Matched hard phrases:", top_matching_phrases(text, hard_lex) or "none")
    st.write("Matched soft phrases:", top_matching_phrases(text, soft_lex) or "none")
    st.session_state["analyzer_export"] = {
        "input_text": text,
        "hard_norm": lex["hard_norm"],
        "soft_norm": lex["soft_norm"],
        "desirability_index": lex["desirability_index"],
        "desirability_class": lex["desirability_class"],
    }

    if model_available:
        x_tfidf = tfidf_obj.transform([clean_text(text)])

        feature_names = np.array(tfidf_obj.get_feature_names_out()) if hasattr(tfidf_obj, "get_feature_names_out") else np.array([])
        if feature_names.size > 0:
            dense = np.asarray(x_tfidf.toarray()).ravel()
            top_idx = dense.argsort()[::-1][:40]
            generic = {"data", "job", "analytics", "role", "work", "using", "use", "like", "know"}
            top_terms = [feature_names[i] for i in top_idx if dense[i] > 0 and feature_names[i] not in generic][:10]
            st.write("Top TF-IDF terms in input:", top_terms if top_terms else "none")

        x_num = np.array([[lex["hard_norm"], lex["soft_norm"], lex["desirability_index"]]], dtype=float)
        need = int(getattr(kmeans_obj, "n_features_in_", x_tfidf.shape[1]))

        if x_tfidf.shape[1] + x_num.shape[1] == need:
            x_pred = hstack([x_tfidf, csr_matrix(x_num)])
        elif x_tfidf.shape[1] == need:
            x_pred = x_tfidf
        elif svd_obj is not None:
            x_svd = svd_obj.transform(x_tfidf)
            if x_svd.shape[1] + x_num.shape[1] == need:
                x_pred = np.hstack([x_svd, x_num])
            else:
                x_pred = x_svd
        else:
            x_pred = x_tfidf

        try:
            pred_cluster = int(kmeans_obj.predict(x_pred)[0])
        except Exception as e:
            st.error(f"Prediction failed due to feature mismatch: {e}")
            st.stop()

        label = f"Cluster {pred_cluster}"
        size_pct = np.nan
        top_terms = "n/a"
        top_phrases = "n/a"

        if not profiles_df.empty and "cluster" in profiles_df.columns:
            m = profiles_df[profiles_df["cluster"] == pred_cluster]
            if not m.empty:
                row = m.iloc[0]
                label = str(row.get("cluster_label", label))
                size_pct = row.get("size_pct", np.nan)
                top_terms = str(row.get("top_tfidf_terms", "n/a"))
                top_phrases = str(row.get("top_weighted_phrases", "n/a"))

        conf_bucket = "Unknown"
        if hasattr(kmeans_obj, "transform"):
            try:
                d = float(np.ravel(kmeans_obj.transform(x_pred))[pred_cluster])
                if d <= 0.8:
                    conf_bucket = "High"
                elif d <= 1.6:
                    conf_bucket = "Medium"
                else:
                    conf_bucket = "Low"
                st.metric("Distance to centroid", f"{d:.3f}", delta=f"Confidence: {conf_bucket}")
            except Exception:
                pass

        st.success(f"Predicted cluster: {pred_cluster} - {label}")
        if pd.notna(size_pct):
            st.write(f"Cluster size share: **{float(size_pct):.1f}%**")
        st.write(f"Cluster top terms: {top_terms}")
        st.write(f"Cluster top phrases: {top_phrases}")

        cdf = filtered[filtered["cluster"] == pred_cluster]
        if not cdf.empty:
            med = float(cdf["comments_count_observed"].median())
            q1 = float(cdf["comments_count_observed"].quantile(0.25))
            q3 = float(cdf["comments_count_observed"].quantile(0.75))
            tone = float(cdf["delta_sentiment"].mean())
            st.info(f"Expected engagement band (median ± IQR): {med:.1f} (Q1={q1:.1f}, Q3={q3:.1f}).")
            st.info(f"Expected community tone shift for this cluster: {tone:+.3f}.")
            st.session_state["analyzer_export"].update(
                {
                    "pred_cluster": pred_cluster,
                    "cluster_label": label,
                    "expected_median_engagement": med,
                    "expected_q1_engagement": q1,
                    "expected_q3_engagement": q3,
                    "expected_delta_sentiment": tone,
                }
            )
    else:
        st.info("Model unavailable: predicted cluster cannot be computed. Use lexicon diagnostics above.")
        st.session_state["analyzer_export"].update({"pred_cluster": None, "cluster_label": "lexicon_only"})

if "analyzer_export" in st.session_state:
    export_df = pd.DataFrame([st.session_state["analyzer_export"]])
    st.download_button(
        "Export analyzer result CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="post_analyzer_result.csv",
        mime="text/csv",
    )

st.subheader("So what?")
st.markdown(
    """
- Diagnostics expose what drove the estimate rather than only the final label.
- Phrase matches and top terms make framing decisions auditable.
- Use this as directional support for review, not as a final automated decision.
"""
)
