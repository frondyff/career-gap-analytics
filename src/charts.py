from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def cluster_distribution_bar(profile_df: pd.DataFrame):
    plot_df = profile_df.sort_values("size_pct", ascending=False)
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y="size_pct",
        color="cluster_label",
        title="Cluster Distribution (% of Posts)",
        labels={"size_pct": "Size %", "cluster_label": "Cluster"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
    return fig


def framing_gauge(hard_value: float, soft_value: float, title: str = "Hard vs Soft framing"):
    total = max(hard_value + soft_value, 1e-9)
    hard_pct = 100 * hard_value / total
    soft_pct = 100 * soft_value / total

    fig = go.Figure(
        data=[
            go.Bar(name="Hard-skill framing", x=[hard_pct], y=[title], orientation="h", marker_color="#0B7285"),
            go.Bar(name="Soft-skill framing", x=[soft_pct], y=[title], orientation="h", marker_color="#F08C00"),
        ]
    )
    fig.update_layout(barmode="stack", xaxis_title="% share", yaxis_title="", height=180, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def desirability_scatter(
    posts_df: pd.DataFrame,
    centroids_only: bool = False,
    use_log_engagement: bool = False,
    use_jitter: bool = False,
    show_density: bool = False,
    random_seed: int = 42,
):
    plot_df = posts_df.copy()
    rng = np.random.default_rng(random_seed)
    if use_jitter and not plot_df.empty:
        jitter = rng.normal(0, 0.08, size=len(plot_df))
        plot_df["engagement_plot"] = plot_df["comments_count_observed"].fillna(0) + jitter
        plot_df["engagement_plot"] = plot_df["engagement_plot"].clip(lower=0)
    else:
        plot_df["engagement_plot"] = plot_df["comments_count_observed"].fillna(0)

    if show_density and not centroids_only:
        fig = px.density_contour(
            plot_df,
            x="desirability_index",
            y="engagement_plot",
            color="cluster_label",
            title="Desirability Index vs Engagement (Density)",
        )
    elif centroids_only:
        cent = (
            plot_df.groupby(["cluster", "cluster_label"], as_index=False)
            .agg(
                desirability_index=("desirability_index", "mean"),
                engagement_plot=("comments_count_observed", "median"),
                post_sentiment=("post_sentiment", "mean"),
                comment_sentiment=("comment_sentiment", "mean"),
                size=("post_id", "size"),
            )
        )
        fig = px.scatter(
            cent,
            x="desirability_index",
            y="engagement_plot",
            size="size",
            color="cluster_label",
            text="cluster_label",
            hover_data=["cluster", "post_sentiment", "comment_sentiment"],
            title="Desirability Index vs Engagement (Cluster Centroids)",
        )
    else:
        fig = px.scatter(
            plot_df,
            x="desirability_index",
            y="engagement_plot",
            color="cluster_label",
            hover_data=["title", "subreddit", "comments_count_observed", "post_sentiment", "comment_sentiment", "desirability_class"],
            title="Desirability Index vs Engagement (Posts)",
            opacity=0.45,
        )

        cent = (
            plot_df.groupby(["cluster", "cluster_label"], as_index=False)
            .agg(desirability_index=("desirability_index", "mean"), engagement_plot=("comments_count_observed", "median"), size=("post_id", "size"))
        )
        if not cent.empty:
            fig.add_trace(
                go.Scatter(
                    x=cent["desirability_index"],
                    y=cent["engagement_plot"],
                    mode="markers+text",
                    text=cent["cluster_label"],
                    textposition="top center",
                    marker=dict(size=14, color="black", symbol="diamond"),
                    name="Centroids",
                )
            )

    if use_log_engagement:
        fig.update_yaxes(type="log", title="Engagement (log scale)")
    else:
        fig.update_yaxes(title="Engagement")

    x_mid = float(plot_df["desirability_index"].median()) if not plot_df.empty else 0.0
    y_mid = float(plot_df["engagement_plot"].median()) if not plot_df.empty else 0.0

    fig.add_vline(x=x_mid, line_dash="dot", line_color="gray")
    fig.add_hline(y=y_mid, line_dash="dot", line_color="gray")

    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text="Top-left: high engagement + soft", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text="Top-right: high engagement + hard", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.02, text="Bottom-left: low engagement + soft", showarrow=False)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.02, text="Bottom-right: low engagement + hard", showarrow=False)

    return fig


def sentiment_delta_bar(posts_df: pd.DataFrame):
    delta = (
        posts_df.groupby(["cluster", "cluster_label"], as_index=False)["delta_sentiment"]
        .mean()
        .sort_values("delta_sentiment", ascending=False)
    )
    fig = px.bar(
        delta,
        x="cluster_label",
        y="delta_sentiment",
        color="cluster_label",
        title="Community Tone Shift by Cluster (Comments - Posts)",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
    fig.add_hline(y=0, line_width=1)
    return fig


def subreddit_stacked_cluster(posts_df: pd.DataFrame, as_percent: bool = False, top_n_clusters: int | None = None):
    g = posts_df.groupby(["subreddit", "cluster_label"], as_index=False).size()
    if top_n_clusters is not None:
        keep = g.groupby("cluster_label")["size"].sum().sort_values(ascending=False).head(top_n_clusters).index
        g = g[g["cluster_label"].isin(keep)]
    if as_percent:
        g["value"] = g["size"]
        g["value"] = g.groupby("subreddit")["value"].transform(lambda s: 100 * s / s.sum())
        y_col = "value"
        y_label = "% of posts"
        title = "Cluster Distribution by Subreddit (%)"
    else:
        y_col = "size"
        y_label = "Post count"
        title = "Cluster Distribution by Subreddit"

    fig = px.bar(g, x="subreddit", y=y_col, color="cluster_label", title=title, barmode="stack", labels={y_col: y_label})
    return fig


def subreddit_metrics(posts_df: pd.DataFrame):
    g = posts_df.groupby("subreddit", as_index=False).agg(
        avg_desirability_index=("desirability_index", "mean"),
        median_engagement=("comments_count_observed", "median"),
        avg_delta_sentiment=("delta_sentiment", "mean"),
    )
    return g
