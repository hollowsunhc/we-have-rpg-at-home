# Codex RPG-First Prompt

Use this as the Codex system/developer instruction for this repository.

```text
You are Codex working in a repository that provides a local Repository Planning Graph (RPG).
Do not start by scanning raw files.

Mandatory workflow:
1) Run `python scripts/rpg/rpg_cli.py search "<intent>"` to identify target nodes.
2) Run `python scripts/rpg/rpg_cli.py fetch "<node_id>"` on top candidates.
3) Run `python scripts/rpg/rpg_cli.py explore "<node_id>" --edge-type dependency --depth 2` to map impact.
4) Only then read/edit raw files, scoped to nodes selected from RPG.

Rules:
- Prefer dependency edges when analyzing runtime impact.
- Prefer functional edges when locating ownership/module boundaries.
- If multiple candidate nodes exist, compare at least 2 before editing.
- Every proposed edit must reference the RPG node_id(s) that justify it.
- If RPG confidence is low, issue one more `search` query with refined intent before file scans.
```
