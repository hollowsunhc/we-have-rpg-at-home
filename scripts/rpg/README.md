# Local RPG (Repository Planning Graph)

This folder provides a minimal local RPG pipeline aligned with your staged plan:

- `rpg_cli.py`: build/query/serve commands.
- `rpg.schema.json`: data contract for `rpg.json`.
- `RPG_FIRST_PROMPT.md`: Codex RPG-first behavior contract.
- `AI_INTEGRATION.md`: Codex-only integration guide.

## What it builds

`python scripts/rpg/rpg_cli.py build --repo . --out .rpg_data --progress-interval 100`

Output artifacts:

- `.rpg_data/rpg.json`: graph snapshot.
- `.rpg_data/rpg.sqlite`: query DB (`nodes`, `edges`, indexed feature search).
- `.rpg_data/semantic_cache.json`: symbol feature cache; reused unless symbol signature changes.

Useful build controls:

- `--max-file-bytes 1500000`: skip deep parsing for very large files.
- `--progress-interval 100`: print periodic progress so long runs are visible.

Node shape:

```jsonc
{
  "id": "unique_node_id",
  "kind": "directory | file | function | class",
  "feature": ["atomic feature phrase"],
  "meta": {
    "path": "relative/path",
    "symbol": "symbol_name_or_null",
    "language": "cpp | javascript | ...",
    "start_line": 1,
    "end_line": 42
  }
}
```

Edge shape:

```jsonc
{
  "from": "node_id",
  "to": "node_id",
  "type": "functional | dependency",
  "relation": "contains | scopes | includes | calls"
}
```

## Query tools

These map directly to the tool surface you described:

- `SearchNode(intent)`:
  - `python scripts/rpg/rpg_cli.py search "skeletonization pipeline" --db .rpg_data/rpg.sqlite`
- `FetchNode(node_id)`:
  - `python scripts/rpg/rpg_cli.py fetch "fn:rxmesh/libs/skeleton/reeb_skeleton.cpp:BuildReebSkeletonResultCPU:25"`
- `ExploreRPG(node_id, edge_type, depth)`:
  - `python scripts/rpg/rpg_cli.py explore "file:rxmesh/libs/skeleton/reeb_skeleton.cpp" --db .rpg_data/rpg.sqlite --edge-type dependency --depth 2`

## Optional local service

Start local HTTP endpoints:

`python scripts/rpg/rpg_cli.py serve --db .rpg_data/rpg.sqlite --host 127.0.0.1 --port 7878`

Endpoints:

- `GET /health`
- `GET /search?intent=...&limit=10`
- `GET /fetch?node_id=...`
- `GET /explore?node_id=...&edge_type=dependency&depth=2&direction=both`

## Notes

- Dependency edges are static and best effort:
  - C/C++ include graph (`#include "..."`)
  - Function call graph from parsed function bodies
- Functional centroids are currently rule-based categories. You can replace this assignment step with an LLM clustering pass later without changing storage/query contracts.
- SQLite is configured with journaling disabled for sandbox compatibility; regenerate the DB from source as needed.
- Non-Codex integration helpers are intentionally disabled in this repo.
