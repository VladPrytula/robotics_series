# Publishing Workflow

This project uses two git remotes with different visibility levels.

## Remotes

| Remote   | URL | Visibility | Content |
|----------|-----|------------|---------|
| `origin` | `ssh://git@codeberg.org/VladP/rl_for_robotics_from_scratch.git` | **Private** | Everything: book materials, personas, proposals, tutorials |
| `github` | `git@github.com:VladPrytula/robotics_series.git` | **Public** | Filtered: tutorials, scripts, training code (no book materials) |

## Day-to-day: Push to Codeberg (private)

Codeberg is the source of truth. All work goes here.

```bash
git add -A
git commit -m "your commit message"
git push origin main
```

This pushes everything -- Manning chapters, personas, proposals, CLAUDE.md, etc.

## Occasionally: Push to GitHub (public)

When you want to update the public repo with the latest tutorials and code:

```bash
bash scripts/push-public.sh
```

This script:
1. Creates a temporary branch from your current HEAD
2. Removes private files (Manning/, manning_proposal/, personas, etc.)
3. Swaps in a public-safe `.gitignore`
4. Force-pushes to `github/main`
5. Returns you to your working branch

To preview what would be pushed without actually pushing:

```bash
bash scripts/push-public.sh --dry-run
```

## What stays private (excluded from GitHub)

These paths are stripped by `push-public.sh` and listed in its `PRIVATE_PATHS` array:

- `Manning/` -- book chapters, reviews, scaffolds
- `manning_proposal/` -- proposal, TOC, personas, protocols
- `vlad_prytula_persona.md` -- author persona
- `CLAUDE.md` -- Claude Code project instructions
- `AGENTS.md` -- agent configuration
- `content/POSTING_NOTES.md` -- posting notes
- `syllabus_book.md` -- book-specific syllabus
- `scripts/push-public.sh` -- the filtering script itself
- `PUBLISHING.md` -- this file

## Adding new private files

If you create a new file or directory that should stay private:

1. Add it to the `PRIVATE_PATHS` array in `scripts/push-public.sh`
2. That's it -- the script handles the rest

The `.gitignore` in this repo tracks everything (for Codeberg).
The script generates a separate public `.gitignore` when pushing to GitHub.

## SSH setup

Each remote uses a separate SSH key:

| Service | Key file | SSH config host |
|---------|----------|-----------------|
| GitHub | `~/.ssh/id_ed25519` | `github.com` |
| Codeberg | `~/.ssh/id_ed25519_codeberg` | `codeberg.org` |

Config lives in `~/.ssh/config`. To test connectivity:

```bash
ssh -T git@github.com       # should greet you
ssh -T git@codeberg.org     # should greet you as VladP
```
