#!/bin/bash
# Cleanup script to remove .env from git history using git-filter-repo or BFG.
# Do NOT run this unless you understand that it rewrites history and requires force-push.

set -e

echo "This script rewrites git history to remove .env and any occurrences of secrets."
echo "Recommended: make a backup of your repo first."

echo "Options:"
echo "1) Use BFG Repo-Cleaner (recommended for ease):"
echo "   java -jar bfg.jar --delete-files .env"
echo "   git reflog expire --expire=now --all && git gc --prune=now --aggressive"
echo "   git push --force"

echo "2) Use git-filter-repo (recommended if installed):"
echo "   git filter-repo --invert-paths --paths .env"
echo "   git push --force"

echo "3) Safer (non-destructive): remove .env from index and commit:
   git rm --cached .env
   git commit -m 'Remove .env from repository'
   git push"

echo "Note: After history rewrite, rotate the exposed API keys in Google Cloud immediately."