from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import compute_phrase_document_frequency, initialize_state, load_hard_lexicon, load_soft_lexicon

TECH_HINTS = {"sql", "python", "etl", "spark", "airflow", "dbt", "cloud", "aws", "gcp", "pipeline", "warehouse"}

st.set_page_config(page_title="Lexicon Transparency", layout="wide")
st.title("Lexicon Transparency")

posts_df, profiles_df, filtered, filters, warnings, _ = initialize_state()
for w in warnings:
    st.warning(w)

hard_lex, hard_path = load_hard_lexicon()
soft_lex, soft_path = load_soft_lexicon()
if not hard_lex and not soft_lex:
    st.warning("Lexicon files are missing. Add hard/soft lexicon JSON files under artifacts_refined.")
    st.stop()

st.caption(f"Loaded hard lexicon: {hard_path or 'n/a'} | soft lexicon: {soft_path or 'n/a'}")

st.markdown(
    """
Turney-style PMI orientation (plain language):
- We extract recurring phrases from posts.
- We compare phrase association with hard-skill seeds and soft-skill seeds.
- Phrase weights represent association strength in this corpus.
- These weights support transparency and interpretation, not causal claims.
"""
)

freq_df = compute_phrase_document_frequency(posts_df)
freq_map = dict(zip(freq_df["phrase"], freq_df["doc_freq"])) if not freq_df.empty else {}

all_weights = [*hard_lex.values(), *soft_lex.values()]
w_min, w_max = (0.0, float(max(all_weights))) if all_weights else (0.0, 1.0)
min_weight = st.slider("Min weight", min_value=0.0, max_value=max(w_max, 1.0), value=0.0, step=0.01)
max_freq = int(freq_df["doc_freq"].max()) if not freq_df.empty else 1
min_freq = st.slider("Min phrase frequency", min_value=0, max_value=max(max_freq, 1), value=0)

hard_df = pd.DataFrame(sorted(hard_lex.items(), key=lambda x: x[1], reverse=True), columns=["phrase", "weight"])
soft_df = pd.DataFrame(sorted(soft_lex.items(), key=lambda x: x[1], reverse=True), columns=["phrase", "weight"])
hard_df["doc_freq"] = hard_df["phrase"].map(lambda p: int(freq_map.get(p, 0)))
soft_df["doc_freq"] = soft_df["phrase"].map(lambda p: int(freq_map.get(p, 0)))

hard_view = hard_df[(hard_df["weight"] >= min_weight) & (hard_df["doc_freq"] >= min_freq)].head(30)
soft_view = soft_df[(soft_df["weight"] >= min_weight) & (soft_df["doc_freq"] >= min_freq)].head(30)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Top hard-skill phrases")
    st.dataframe(hard_view, use_container_width=True)
with c2:
    st.subheader("Top soft-skill phrases")
    st.dataframe(soft_view, use_container_width=True)

st.download_button("Export filtered lexicon view CSV", pd.concat([hard_view.assign(lexicon="hard"), soft_view.assign(lexicon="soft")]).to_csv(index=False).encode("utf-8"), "lexicon_filtered_view.csv", "text/csv")

# Sanity check table
st.subheader("Sanity check (needs review)")
soft_df["technical_hint"] = soft_df["phrase"].apply(lambda p: any(t in p.lower() for t in TECH_HINTS))
needs_review = soft_df[soft_df["technical_hint"]].head(10).copy()
if needs_review.empty:
    st.info("No obvious technical phrases detected in the current soft lexicon top set.")
else:
    needs_review["review_note"] = "Needs review (technical phrase appears in soft lexicon)"
    st.dataframe(needs_review[["phrase", "weight", "doc_freq", "review_note"]], use_container_width=True)

# phrase search and snippets
st.subheader("Phrase search with evidence")
q = st.text_input("Search phrase", value="")
if q.strip():
    ql = q.strip().lower()
    hw = float(hard_lex.get(ql, 0.0))
    sw = float(soft_lex.get(ql, 0.0))
    orient = hw - sw
    if hw == 0 and sw == 0:
        st.info("Phrase not found in either lexicon.")
    else:
        where = "hard" if hw > sw else ("soft" if sw > hw else "both")
        sign = "positive (hard)" if orient > 0 else ("negative (soft)" if orient < 0 else "neutral")
        st.write(pd.DataFrame([{"phrase": ql, "orientation_sign": sign, "orientation": orient, "hard_weight": hw, "soft_weight": sw, "lexicon": where, "doc_freq": int(freq_map.get(ql, 0))}]))

        blob = (posts_df["title"].fillna("") + " " + posts_df["body"].fillna("") + " " + posts_df["comments_text"].fillna("")).str.lower()
        hits = posts_df[blob.str.contains(ql, na=False)].head(2)
        if hits.empty:
            st.caption("No example snippets found in current data source.")
        else:
            st.write("Example snippets:")
            for _, row in hits.iterrows():
                snippet = (str(row.get("title", "")) + " " + str(row.get("body", "")) + " " + str(row.get("comments_text", "")))
                st.markdown(f"- **{row.get('subreddit','n/a')}**: {snippet[:260]}...")

st.subheader("So what?")
st.markdown(
    """
- Weight and frequency filters help separate robust phrases from noisy tail phrases.
- The needs-review table surfaces ambiguous assignments for analyst validation.
- Phrase-level snippets improve trust by linking lexicon entries back to observed text.
"""
)
