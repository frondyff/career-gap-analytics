from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts_refined"
FIGURES = ROOT / "figures"
SAMPLE_DATA = ROOT / "sample_data"
MODELS_DIR = ARTIFACTS / "models"

# Preferred file names, with fallbacks in order.
ENRICHED_CANDIDATES = [
    ARTIFACTS / "final_enriched_posts.parquet",
    ARTIFACTS / "final_enriched_posts.csv",
    ARTIFACTS / "final_posts_enriched.parquet",
    ARTIFACTS / "final_posts_enriched.csv",
    ARTIFACTS / "final_posts_enriched_refined.parquet",
    ARTIFACTS / "final_posts_enriched_refined.csv",
]

PROFILE_CANDIDATES = [
    ARTIFACTS / "cluster_profiles.csv",
    ARTIFACTS / "cluster_profile_refined.csv",
    ARTIFACTS / "cluster_summary_table.csv",
    ARTIFACTS / "tables/cluster_profile_refined.csv",
]

HARD_LEXICON_CANDIDATES = [
    ARTIFACTS / "hard_lexicon.json",
    ARTIFACTS / "lexicons/hard_pmi_weighted.json",
    ARTIFACTS / "hard.json",
]

SOFT_LEXICON_CANDIDATES = [
    ARTIFACTS / "soft_lexicon.json",
    ARTIFACTS / "lexicons/soft_pmi_weighted.json",
    ARTIFACTS / "soft.json",
]

MODEL_CANDIDATES = {
    "tfidf": [
        ARTIFACTS / "models/tfidf_vectorizer.joblib",
        ARTIFACTS / "tfidf_vectorizer.joblib",
        ARTIFACTS / "models/refined_tfidf_vectorizer.joblib",
    ],
    "kmeans": [
        ARTIFACTS / "models/kmeans_model.joblib",
        ARTIFACTS / "kmeans_model.joblib",
        ARTIFACTS / "models/refined_clustering_model.joblib",
    ],
    "svd": [
        ARTIFACTS / "models/svd_model.joblib",
        ARTIFACTS / "models/refined_svd_model.joblib",
    ],
}

REQUIRED_POST_COLUMNS = [
    "post_id",
    "subreddit",
    "cluster",
    "post_sentiment",
    "comment_sentiment",
    "delta_sentiment",
    "comments_count_observed",
    "hard_norm",
    "soft_norm",
    "desirability_index",
    "desirability_class",
]

REQUIRED_PROFILE_COLUMNS = [
    "cluster",
    "size",
    "size_pct",
    "top_tfidf_terms",
    "top_weighted_phrases",
    "avg_sent_post",
    "avg_sent_comment",
    "avg_hard_norm",
    "avg_soft_norm",
    "dominant_class",
]

DESIRABILITY_CLASSES = ["hard_dominant", "soft_dominant", "balanced"]
