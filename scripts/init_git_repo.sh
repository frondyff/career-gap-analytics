#!/usr/bin/env bash
set -euo pipefail

git init
git add .
git commit -m "Initial commit: INSY669 Streamlit app"

echo "\nRepository initialized. Next steps:\n"
echo "Option A (GitHub CLI):"
echo "  gh repo create <REPO_NAME> --public --source=. --remote=origin --push"
echo "\nOption B (manual):"
echo "  1) Create an empty repo on GitHub"
echo "  2) git remote add origin <REPO_URL>"
echo "  3) git branch -M main"
echo "  4) git push -u origin main"
