from __future__ import annotations

import streamlit as st

from src.charts import desirability_scatter
from src.data import active_filters_text, compute_baseline_stats, framing_interpretation, initialize_state, sentiment_delta_label

st.set_page_config(page_title="Desirability Spectrum", layout="wide")
st.title("Desirability Spectrum")

posts_df, profiles_df, filtered, filters, warnings, _ = initialize_state()
for w in warnings:
    st.warning(w)
if posts_df.empty:
    st.error("No usable data loaded.")
    st.stop()
if filtered.empty:
    st.warning("No rows match current filters.")
    st.stop()

st.caption(active_filters_text(filters))
baseline_df = filtered if filters.get("baseline_mode") == "filtered" else posts_df
baseline = compute_baseline_stats(baseline_df)

c1, c2, c3, c4 = st.columns(4)
centroids_only = c1.toggle("Centroids only", value=False)
use_log = c2.toggle("Log-scale engagement", value=False)
use_jitter = c3.toggle("Jitter points", value=True)
show_density = c4.toggle("Density view", value=False)

fig = desirability_scatter(
    filtered,
    centroids_only=centroids_only,
    use_log_engagement=use_log,
    use_jitter=use_jitter,
    show_density=show_density,
)
st.plotly_chart(fig, use_container_width=True)

avg_idx = filtered["desirability_index"].mean()
avg_delta = filtered["delta_sentiment"].mean()

st.metric("Avg desirability index", f"{avg_idx:.3f}", delta=f"{avg_idx-baseline['desirability_index']:+.3f}")
st.metric("Avg delta sentiment", f"{avg_delta:.3f}", delta=f"{avg_delta-baseline['delta_sentiment']:+.3f}")
st.caption(f"{sentiment_delta_label(avg_delta)}. Sentiment is heuristic and Reddit-specific.")

st.download_button(
    "Export current spectrum data CSV",
    filtered[[c for c in ["post_id", "cluster", "cluster_label", "subreddit", "desirability_index", "comments_count_observed", "post_sentiment", "comment_sentiment", "delta_sentiment", "desirability_class"] if c in filtered.columns]].to_csv(index=False).encode("utf-8"),
    "desirability_spectrum_slice.csv",
    "text/csv",
)

st.subheader("So what?")
hard_share = (filtered["desirability_index"] > 0).mean() * 100
high_eng = filtered["comments_count_observed"].quantile(0.75)
st.markdown(
    f"""
- **{hard_share:.1f}%** of filtered posts lean hard-skill in framing.
- Top engagement quartile starts near **{high_eng:.1f}** observed comments.
- Overall framing reads as **{framing_interpretation(avg_idx, baseline['desirability_index'])}**.
"""
)
