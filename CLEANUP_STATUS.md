# Cleanup Status

This branch is an additive reviewer-safety cleanup branch based on `moe-routing/main` at:

```text
8fff8a71 Sanitize reviewer references
```

## Non-Disruption Rules

- No force-pushes.
- No branch deletion.
- No movement or deletion of existing experiment folders.
- No rewrite of historical evidence files.
- Add navigation and policy docs first; defer physical artifact pruning until reviewers are no longer relying on the current tree.

## Current Remote

Local remote name:

```text
moe-routing
```

Remote URL:

```text
git@github.com:jeffreywilliamportfolio/moe-routing.git
```

The requested spelling `jeffreywilliamwork/moe-routing` does not match the configured local remote in this checkout.

## Recommended Next Cleanup After Review

After active review is complete, the next low-risk cleanup is to open a separate pruning branch that removes tracked generated caches and raw tensor files from `main` while keeping frozen branches available for provenance.

Likely candidates:

- tracked `__pycache__/` files
- tracked `.npy` tensor dumps in older non-current folders
- duplicated generated output folders that are already summarized by committed JSON/Markdown

That pruning should be reviewed as a separate diff because it will be large and noisy.
