# Codex Integration Only

This repository is intentionally configured for Codex-only RPG usage.

## 1) Build and refresh the RPG

```powershell
python scripts/rpg/rpg_cli.py build --repo . --progress-interval 100
```

Re-run after meaningful code changes.

## 2) Use Codex with RPG-first commands

Codex should query RPG before reading or editing raw files:

- `SearchNode(intent)`:
  - `python scripts/rpg/rpg_cli.py search "skeletonization pipeline"`
- `FetchNode(node_id)`:
  - `python scripts/rpg/rpg_cli.py fetch "fn:rxmesh/libs/skeleton/reeb_skeleton.cpp:BuildReebSkeletonResultCPU:137"`
- `ExploreRPG(node_id, edge_type, depth)`:
  - `python scripts/rpg/rpg_cli.py explore "file:rxmesh/libs/skeleton/reeb_skeleton.cpp" --edge-type dependency --depth 2`

## 3) Optional Codex HTTP endpoint

If needed, run the local service:

```powershell
python scripts/rpg/rpg_cli.py serve --db .rpg_data/rpg.sqlite --host 127.0.0.1 --port 7878
```

Endpoints:

- `GET /search?intent=...&limit=10`
- `GET /fetch?node_id=...`
- `GET /explore?node_id=...&edge_type=dependency&depth=2&direction=both`

## 4) Codex operating pattern

1. Run `search`.
2. Validate candidates with `fetch` and `explore`.
3. Edit only files implied by selected node metadata.
4. Re-run `explore` to check dependency blast radius.
