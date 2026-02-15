$ErrorActionPreference = "Stop"

git init
git add .
git commit -m "Initial commit: INSY669 Streamlit app"

Write-Host ""
Write-Host "Repository initialized. Next steps:"
Write-Host ""
Write-Host "Option A (GitHub CLI):"
Write-Host "  gh repo create <REPO_NAME> --public --source=. --remote=origin --push"
Write-Host ""
Write-Host "Option B (manual):"
Write-Host "  1) Create an empty repo on GitHub"
Write-Host "  2) git remote add origin <REPO_URL>"
Write-Host "  3) git branch -M main"
Write-Host "  4) git push -u origin main"
