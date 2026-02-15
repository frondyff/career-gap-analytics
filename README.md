# INSY669 Text Analytics Streamlit App

## Project Summary
This Streamlit app provides a stakeholder-friendly interface for exploring Reddit discussions about data analytics careers. It helps users:
- browse cluster themes with interpretable labels,
- inspect hard-skill vs soft-skill desirability framing,
- compare post vs comment sentiment and engagement,
- review representative posts and subcommunity differences.

## Run Locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Modes in App
At the top of the app sidebar, choose one mode:
- `Demo (sample data)` uses `sample_data/`.
- `Load artifacts from local repo` loads `artifacts_refined/` files.
- `Upload artifacts` lets you upload enriched posts and cluster profile files directly.

## Expected `artifacts_refined/` Files
Primary expected files:
- `final_enriched_posts.parquet` or `.csv`
- `cluster_profiles.csv`
- `hard_lexicon.json`
- `soft_lexicon.json`

The app also supports current fallback names (for compatibility), such as:
- `final_posts_enriched.parquet`
- `cluster_summary_table.csv`
- lexicons under `artifacts_refined/lexicons/`

Optional model files for Post Analyzer:
- `models/tfidf_vectorizer.joblib`
- `models/kmeans_model.joblib`
- `models/svd_model.joblib` (optional)

If optional files are missing, the corresponding feature is hidden with a warning.

## Generate Artifacts from Notebook
Run your refined notebook pipeline first, then ensure artifacts are saved under `artifacts_refined/`.
Typical command:
```bash
jupyter notebook
# Open INSY669_Text_Analytics_Research_Grade.ipynb and run all cells
```

## Repo Structure
```text
.
├── app.py
├── pages/
│   ├── 0_Executive_Summary.py
│   ├── 1_Cluster_Explorer.py
│   ├── 2_Desirability_Spectrum.py
│   ├── 3_Post_Analyzer.py
│   ├── 4_Lexicon_Transparency.py
│   └── 5_Subcommunity_Compare.py
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── charts.py
│   ├── labels.py
│   └── text_utils.py
├── sample_data/
│   ├── sample_final_enriched_posts.csv
│   └── sample_cluster_profiles.csv
├── scripts/
│   ├── init_git_repo.sh
│   └── init_git_repo.ps1
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── requirements.txt
├── runtime.txt
├── packages.txt
└── README.md
```

## Push to GitHub
### Option A: GitHub CLI
```bash
gh repo create <REPO_NAME> --public --source=. --remote=origin --push
```

### Option B: Manual
```bash
git init
git add .
git commit -m "Initial commit: INSY669 Streamlit app"
git remote add origin <REPO_URL>
git branch -M main
git push -u origin main
```

If GitHub CLI is not installed, use Option B.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud > New app.
3. Select repository/branch and set app path to `app.py`.
4. Add secrets only if needed (not required by default).
5. Deploy and verify pages, filters, and charts.
6. If dependencies fail, confirm `requirements.txt` and `runtime.txt` are present in the repo root.

## Streamlit Cloud Notes
- Large local artifacts may not be available in cloud containers.
- Use `Demo (sample data)` mode or `Upload artifacts` mode for cloud demos.
- `sample_data/` is intentionally small to keep deployment lightweight.
- If local artifacts are missing, the app auto-switches local mode to demo mode and shows a warning banner.

## Screenshots (Placeholder)
- `docs/screenshots/overview.png`
- `docs/screenshots/cluster_explorer.png`
- `docs/screenshots/desirability_spectrum.png`

## License
MIT (see `LICENSE`).
