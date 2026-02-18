#!/usr/bin/env bash
set -euo pipefail
# ─────────────────────────────────────────────────────────────
# push-public.sh -- Push filtered content to the public GitHub repo
#
# Usage:  bash scripts/push-public.sh [--dry-run] [-m "commit message"]
#
# What it does:
#   1. Creates a temporary orphan branch from current HEAD
#   2. Removes private files (Manning materials, personas, etc.)
#   3. Removes .gitignore from the public commit
#   4. Force-pushes to the 'github' remote as 'main'
#   5. Returns you to your original branch
#
# The Codeberg repo (origin) is unaffected.
# ─────────────────────────────────────────────────────────────

DRY_RUN=false
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --message|-m) COMMIT_MSG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$COMMIT_MSG" ]]; then
    COMMIT_MSG="public: filtered export from $(git rev-parse --short HEAD)"
fi

GITHUB_REMOTE="github"
PUBLIC_BRANCH="__public_staging"
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Files/dirs to EXCLUDE from the public GitHub repo
PRIVATE_PATHS=(
    "Manning/"
    "manning_proposal/"
    "vlad_prytula_persona.md"
    "content/POSTING_NOTES.md"
    "AGENTS.md"
    "CLAUDE.md"
    "syllabus_book.md"
    "scripts/push-public.sh"
    "PUBLISHING.md"
)

echo "==> Creating temporary branch '${PUBLIC_BRANCH}' from '${ORIGINAL_BRANCH}'..."
git checkout -B "${PUBLIC_BRANCH}" "${ORIGINAL_BRANCH}"

echo "==> Removing private files from index..."
for path in "${PRIVATE_PATHS[@]}"; do
    if git ls-files --error-unmatch "${path}" &>/dev/null; then
        git rm -rf --cached "${path}" >/dev/null 2>&1
        echo "    removed: ${path}"
    else
        echo "    skipped (not tracked): ${path}"
    fi
done

# Remove .gitignore from the public commit (not needed in public repo)
echo "==> Removing .gitignore from index..."
git rm --cached .gitignore >/dev/null 2>&1 || true

echo "==> Committing filtered state..."
git commit --allow-empty -m "${COMMIT_MSG}"

if $DRY_RUN; then
    echo "==> [DRY RUN] Would push to '${GITHUB_REMOTE}/main'"
    echo "    Files that would be pushed:"
    git ls-files | head -50
    echo "    ..."
else
    echo "==> Pushing to '${GITHUB_REMOTE}/main'..."
    git push "${GITHUB_REMOTE}" "${PUBLIC_BRANCH}:main" --force
    echo "==> Done. GitHub updated."
fi

echo "==> Returning to '${ORIGINAL_BRANCH}'..."
git checkout -f "${ORIGINAL_BRANCH}"
git branch -D "${PUBLIC_BRANCH}"

echo "==> Clean. You're back on '${ORIGINAL_BRANCH}'."
