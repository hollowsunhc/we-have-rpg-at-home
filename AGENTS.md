# Codex Repository Instructions

This repository is configured for **Codex-only** usage with a local Repository Planning Graph (RPG).

## Required Workflow (RPG-first)

For code navigation, planning, and edits, Codex should use RPG before raw file scanning.

1. Build or refresh RPG when needed:
   - `python scripts/rpg/rpg_cli.py build --repo . --progress-interval 100`
2. Locate relevant scope with:
   - `python scripts/rpg/rpg_cli.py search "<intent>"`
3. Inspect candidates with:
   - `python scripts/rpg/rpg_cli.py fetch "<node_id>"`
4. Analyze impact with:
   - `python scripts/rpg/rpg_cli.py explore "<node_id>" --edge-type dependency --depth 2`
5. Only then read/edit raw files, scoped to selected RPG nodes.

## Rules

- Prefer `dependency` edges for runtime impact analysis.
- Prefer `functional` edges for ownership/module boundaries.
- Compare at least 2 candidate nodes when ambiguity exists.
- If RPG confidence is low, run another refined `search` before broad file scans.
- Keep edits scoped to files implied by selected node metadata.
