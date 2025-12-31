Hiding credentials and removing accidentally committed secrets

Important: Never commit real API keys or secrets to GitHub.

1. Keep secrets out of the repo

- Add `.env` to `.gitignore`. This repo already includes a `.gitignore` that ignores `.env`.
- Commit `.env.example` with placeholder values instead.

2. Local workflow

- Create a `.env` file locally (do NOT commit it):

  GEMINI_API_KEY=your_real_api_key_here

- Load environment variables locally (we use python-dotenv or load directly in code).

3. Deployments and hosting

- Streamlit Cloud: open your app's settings → Secrets → add `GEMINI_API_KEY`.
- GitHub Actions: set repository secret `GEMINI_API_KEY` and reference it in your workflow.
- Heroku: `heroku config:set GEMINI_API_KEY=...`.

4. If you accidentally committed a secret

- Rotate the exposed secret immediately in the provider console (Google Cloud -> Credentials).
- Remove the secret from git history. Example (use carefully):

  git rm --cached .env
  git commit -m "Remove .env"

  # Use BFG or git-filter-repo for safer removal across history

  # Example using git filter-branch (dangerous on large repos):

  git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env" --prune-empty --tag-name-filter cat -- --all
  git push origin --force --all

- Prefer `git-filter-repo` or BFG Repo-Cleaner instead of filter-branch for modern repos.

5. Verify

- Run `git log --all -- .env` to ensure it no longer appears in history.

6. Rotate keys

- After removal, rotate (recreate) keys in your provider to ensure the old key is invalid.

7. Additional recommendations

- Use short-lived credentials where possible.
- Store secrets in a secret manager (GCP Secret Manager, AWS Secrets Manager) for production.

If you'd like, I can:

- Create or update a GitHub Actions workflow that reads the secret from `secrets.GEMINI_API_KEY`.
- Generate a `.env.example` file (already created).
- Provide exact Streamlit Cloud steps or a script to set secrets.
