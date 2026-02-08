#!/usr/bin/env python3
"""Local Repository Planning Graph (RPG) builder and query service."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

SOURCE_EXTENSIONS = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".inc": "cpp",
    ".js": "javascript",
    ".ts": "typescript",
    ".wgsl": "wgsl",
    ".fbs": "flatbuffers",
    ".cmake": "cmake",
}

KNOWN_BUILD_FILENAMES = {"CMakeLists.txt": "cmake"}

DEFAULT_EXCLUDE_DIRS = {".git", ".vs", ".vscode", "out", "build", "third_party"}

FEATURE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

VERB_HINTS = {
    "add",
    "allocate",
    "append",
    "apply",
    "bind",
    "build",
    "calculate",
    "check",
    "collect",
    "compute",
    "configure",
    "copy",
    "create",
    "decode",
    "deserialize",
    "destroy",
    "dispatch",
    "draw",
    "encode",
    "evaluate",
    "fetch",
    "find",
    "generate",
    "get",
    "init",
    "initialize",
    "load",
    "merge",
    "normalize",
    "parse",
    "prepare",
    "process",
    "read",
    "reduce",
    "release",
    "remove",
    "render",
    "resolve",
    "run",
    "save",
    "score",
    "search",
    "serialize",
    "set",
    "sort",
    "split",
    "transform",
    "update",
    "upload",
    "validate",
    "write",
}

FUNCTION_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "decltype",
    "new",
    "delete",
    "else",
    "case",
    "defined",
    "comment",
    "include",
    "pragma",
    "error",
    "warning",
}

CATEGORY_RULES = [
    ("Geometry Processing", ("mesh", "skeleton", "reeb", "geometry", "geodesic", "surface", "topology")),
    ("Rendering & Runtime", ("renderer", "render", "backend", "vulkan", "webgpu", "viewer", "shader", "app", "web")),
    ("IO", ("io", "serialization", "deserialize", "serialize", "asset", "flatbuffer", "schema")),
    ("Build & Tooling", ("cmake", "script", "scripts", "tool", "preset", "config")),
    ("Tests", ("test", "tests", "doctest", "benchmark")),
]
DEFAULT_CATEGORY = "Utilities"


@dataclass
class Node:
    id: str
    kind: str
    feature: list[str]
    path: str | None = None
    symbol: str | None = None
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "feature": self.feature,
            "meta": {
                "path": self.path,
                "symbol": self.symbol,
                "language": self.language,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
        }


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    type: str
    relation: str

    def to_json(self) -> dict[str, str]:
        return {"from": self.from_id, "to": self.to_id, "type": self.type, "relation": self.relation}


@dataclass
class SymbolRecord:
    kind: str
    name: str
    start_line: int
    end_line: int
    signature_hash: str
    body_text: str | None = None


CLASS_DECL_RE = re.compile(r"^\s*(?:template\s*<[^>]*>\s*)?(?:class|struct)\s+([A-Za-z_]\w*)\b")
CPP_FUNC_NAME_RE = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*$")
JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(",
    re.MULTILINE,
)
JS_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b", re.MULTILINE)
JS_METHOD_RE = re.compile(r"^\s*(?:async\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{", re.MULTILINE)
CPP_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
CALL_TOKEN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

MAX_FILE_BYTES_DEFAULT = 1_500_000
MAX_SIGNATURE_SPAN = 1200
MAX_BODY_SCAN_AHEAD = 1200
MAX_SYMBOLS_PER_FILE = 500


def to_posix_rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def split_identifier(name: str) -> list[str]:
    token = name.split("::")[-1]
    token = re.sub(r"[^A-Za-z0-9_]+", " ", token)
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
    words = [w.lower() for w in token.replace("_", " ").split()]
    return [w for w in words if w and w not in FEATURE_STOP_WORDS]


def normalize_text_words(text: str) -> list[str]:
    text = re.sub(r"[^A-Za-z0-9_]+", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = [w.lower() for w in text.split()]
    return [w for w in words if w and w not in FEATURE_STOP_WORDS]


def feature_from_words(words: list[str]) -> str:
    if not words:
        return "handle symbol"
    if words[0] in VERB_HINTS and len(words) > 1:
        return f"{words[0]} {' '.join(words[1:])}"
    if len(words) == 1:
        return f"handle {words[0]}"
    return f"handle {' '.join(words)}"


def infer_category(path: str) -> str:
    low = path.lower()
    for category, needles in CATEGORY_RULES:
        if any(needle in low for needle in needles):
            return category
    return DEFAULT_CATEGORY


def make_directory_features(rel_dir: str) -> list[str]:
    if rel_dir in ("", "."):
        return ["organize repository root"]
    words = normalize_text_words(PurePosixPath(rel_dir).name)
    return uniq([feature_from_words(words), f"organize {PurePosixPath(rel_dir).name.lower()} modules"])


def make_file_features(rel_path: str) -> list[str]:
    p = PurePosixPath(rel_path)
    stem_words = normalize_text_words(p.stem)
    parent_words = normalize_text_words(p.parent.name) if p.parent.name else []
    category = infer_category(rel_path).lower()
    out = [feature_from_words(stem_words), f"support {category} workflows"]
    if parent_words:
        out.append(f"support {' '.join(parent_words)} components")
    return uniq(out)


def make_symbol_features(
    symbol_name: str, rel_path: str, kind: str, semantic_cache: dict[str, object], signature_hash: str
) -> list[str]:
    cache_key = f"{rel_path}:{kind}:{symbol_name}"
    cached = semantic_cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("hash") == signature_hash:
        cached_features = cached.get("features")
        if isinstance(cached_features, list) and cached_features:
            return [str(item) for item in cached_features]
    words = split_identifier(symbol_name)
    category = infer_category(rel_path).lower()
    out = [feature_from_words(words), f"support {category} logic"]
    out.append("encapsulate domain behavior" if kind == "class" else "implement executable operation")
    out = uniq(out)
    semantic_cache[cache_key] = {"hash": signature_hash, "features": out}
    return out


def build_line_starts(text: str) -> list[int]:
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def line_from_offset(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def hash_text(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def mask_cpp_comments_and_strings(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    in_line_comment = False
    in_block_comment = False
    in_string: str | None = None
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                out.extend([" ", " "])
                in_block_comment = False
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if in_string:
            if ch == "\\":
                out.append(" ")
                if i + 1 < n:
                    out.append(" " if text[i + 1] != "\n" else "\n")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            out.extend([" ", " "])
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            out.extend([" ", " "])
            in_block_comment = True
            i += 2
            continue
        if ch == '"' or ch == "'":
            out.append(" ")
            in_string = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def find_matching_delim(text: str, open_index: int, open_ch: str, close_ch: str) -> int | None:
    depth = 0
    for i in range(open_index, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def find_statement_start(masked: str, left_index: int) -> int:
    candidates = [
        masked.rfind("\n", 0, left_index),
        masked.rfind(";", 0, left_index),
        masked.rfind("{", 0, left_index),
        masked.rfind("}", 0, left_index),
        masked.rfind("#", 0, left_index),
    ]
    return max(candidates) + 1


def consume_cpp_postfix(masked: str, start: int) -> tuple[str | None, int | None]:
    i = start
    n = len(masked)
    limit = min(n, start + MAX_BODY_SCAN_AHEAD)
    while i < limit:
        ch = masked[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if masked.startswith("const", i):
            j = i + 5
            if j == n or not masked[j].isalnum():
                i = j
                continue
        if masked.startswith("noexcept", i):
            i += len("noexcept")
            while i < limit and masked[i].isspace():
                i += 1
            if i < limit and masked[i] == "(":
                close_idx = find_matching_delim(masked, i, "(", ")")
                if close_idx is None or close_idx >= limit:
                    return None, None
                i = close_idx + 1
            continue
        if masked.startswith("requires", i):
            i += len("requires")
            continue
        if ch == ":":
            i += 1
            continue
        if ch == "-" and i + 1 < limit and masked[i + 1] == ">":
            i += 2
            continue
        if ch == "{":
            return "{", i
        if ch == ";":
            return ";", i
        if ch == "=":
            semi = masked.find(";", i, limit)
            if semi < 0:
                return None, None
            return ";", semi
        i += 1
    return None, None


def looks_like_cpp_function_context(masked: str, name_start: int, paren_open: int) -> bool:
    stmt_start = find_statement_start(masked, name_start)
    sig_head = masked[stmt_start:paren_open]
    if len(sig_head) > MAX_SIGNATURE_SPAN:
        return False
    stripped = sig_head.strip()
    if not stripped:
        return False
    if any(ch in sig_head for ch in (".", "=", "?", "!", "[", "]")):
        return False
    if "->" in sig_head:
        return False
    if "(" in sig_head or ")" in sig_head:
        return False
    if "," in sig_head:
        return False
    if re.search(r"\breturn\b", sig_head):
        return False
    tail = stripped.split()[-1]
    if tail in FUNCTION_KEYWORDS:
        return False
    if "typedef" in sig_head or "using " in sig_head:
        return False
    if stripped.endswith("::"):
        return False
    word_tokens = re.findall(r"[A-Za-z_]\w*", sig_head)
    if not word_tokens:
        return False
    return True


def extract_cpp_symbols(text: str, allow_declarations: bool) -> list[SymbolRecord]:
    symbols: list[SymbolRecord] = []
    line_starts = build_line_starts(text)
    masked = mask_cpp_comments_and_strings(text)

    for match in CLASS_DECL_RE.finditer(masked):
        name = match.group(1)
        start_line = line_from_offset(line_starts, match.start(1))
        sig_start = masked.rfind("\n", 0, match.start()) + 1
        sig_end = masked.find("\n", match.end())
        if sig_end < 0:
            sig_end = len(masked)
        signature = text[sig_start:sig_end]
        signature_hash = hash_text(signature)
        end_line = start_line
        brace_pos = masked.find("{", match.end())
        if brace_pos >= 0:
            close_idx = find_matching_delim(masked, brace_pos, "{", "}")
            if close_idx is not None:
                end_line = line_from_offset(line_starts, close_idx)
        symbols.append(SymbolRecord("class", name, start_line, end_line, signature_hash))

    seen: set[tuple[str, int]] = set()
    i = 0
    n = len(masked)
    while i < n and len(symbols) < MAX_SYMBOLS_PER_FILE:
        paren_open = masked.find("(", i)
        if paren_open < 0:
            break
        j = paren_open - 1
        while j >= 0 and masked[j].isspace():
            j -= 1
        if j < 0:
            i = paren_open + 1
            continue
        name_end = j + 1
        while j >= 0 and (masked[j].isalnum() or masked[j] == "_" or masked[j] == ":"):
            j -= 1
        name = masked[j + 1 : name_end]
        if not name or not CPP_FUNC_NAME_RE.fullmatch(name):
            i = paren_open + 1
            continue
        short_name = name.split("::")[-1]
        if short_name in FUNCTION_KEYWORDS:
            i = paren_open + 1
            continue
        if not looks_like_cpp_function_context(masked, j + 1, paren_open):
            i = paren_open + 1
            continue

        paren_close = find_matching_delim(masked, paren_open, "(", ")")
        if paren_close is None:
            i = paren_open + 1
            continue
        term, term_idx = consume_cpp_postfix(masked, paren_close + 1)
        if term is None or term_idx is None:
            i = paren_open + 1
            continue
        if term == ";" and not allow_declarations:
            i = paren_open + 1
            continue

        stmt_start = find_statement_start(masked, j + 1)
        signature = text[stmt_start : term_idx + 1].strip()
        if not signature:
            i = paren_open + 1
            continue

        start_line = line_from_offset(line_starts, j + 1)
        end_line = start_line
        body_text = None
        if term == "{":
            close_idx = find_matching_delim(masked, term_idx, "{", "}")
            if close_idx is not None:
                end_line = line_from_offset(line_starts, close_idx)
                body_text = text[term_idx + 1 : close_idx]

        dedupe_key = (name, start_line)
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            symbols.append(
                SymbolRecord(
                    kind="function",
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    signature_hash=hash_text(signature),
                    body_text=body_text,
                )
            )
        i = paren_open + 1

    return symbols


def extract_js_symbols(text: str) -> list[SymbolRecord]:
    symbols: list[SymbolRecord] = []
    line_starts = build_line_starts(text)
    seen: set[tuple[str, int]] = set()

    for match in JS_CLASS_RE.finditer(text):
        name = match.group(1)
        line = line_from_offset(line_starts, match.start(1))
        symbols.append(SymbolRecord("class", name, line, line, hash_text(match.group(0))))

    for match in JS_FUNCTION_RE.finditer(text):
        name = match.group(1)
        line = line_from_offset(line_starts, match.start(1))
        symbols.append(SymbolRecord("function", name, line, line, hash_text(match.group(0))))
        seen.add((name, line))

    for match in JS_METHOD_RE.finditer(text):
        name = match.group(1)
        line = line_from_offset(line_starts, match.start(1))
        key = (name, line)
        if key in seen:
            continue
        symbols.append(SymbolRecord("function", name, line, line, hash_text(match.group(0))))
    return symbols


def parse_symbols_for_file(language: str, text: str, rel_path: str) -> list[SymbolRecord]:
    if language in ("cpp", "c"):
        ext = PurePosixPath(rel_path).suffix.lower()
        allow_declarations = ext in (".h", ".hh", ".hpp", ".hxx", ".inc")
        return extract_cpp_symbols(text, allow_declarations=allow_declarations)
    if language in ("javascript", "typescript"):
        return extract_js_symbols(text)
    return []


def parse_cpp_includes(text: str) -> list[str]:
    return [m.group(1).strip() for m in CPP_INCLUDE_RE.finditer(text)]


def normalize_rel_path(path: str) -> str:
    return str(PurePosixPath(path))


def resolve_include_path(
    current_file: str,
    include_target: str,
    all_files: set[str],
    basename_to_paths: dict[str, list[str]],
) -> str | None:
    include_norm = normalize_rel_path(include_target)
    current_parent = str(PurePosixPath(current_file).parent)
    candidate_rel = normalize_rel_path(str(PurePosixPath(current_parent) / include_norm))
    if candidate_rel in all_files:
        return candidate_rel
    if include_norm in all_files:
        return include_norm
    basename = PurePosixPath(include_norm).name
    options = basename_to_paths.get(basename, [])
    if len(options) == 1:
        return options[0]
    suffix_match = [p for p in options if p.endswith(include_norm)]
    if len(suffix_match) == 1:
        return suffix_match[0]
    return None


def collect_source_files(repo_root: Path, exclude_dirs: set[str]) -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        root_path = Path(root)
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in SOURCE_EXTENSIONS or filename in KNOWN_BUILD_FILENAMES:
                files.append(root_path / filename)
    files.sort()
    return files


def language_for_file(path: Path) -> str | None:
    if path.name in KNOWN_BUILD_FILENAMES:
        return KNOWN_BUILD_FILENAMES[path.name]
    return SOURCE_EXTENSIONS.get(path.suffix.lower())


def read_semantic_cache(cache_path: Path) -> dict[str, object]:
    if not cache_path.exists():
        return {}
    try:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}


def write_semantic_cache(cache_path: Path, cache: dict[str, object]) -> None:
    cache_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def add_edge(
    edge_set: set[tuple[str, str, str, str]],
    edges: list[Edge],
    from_id: str,
    to_id: str,
    edge_type: str,
    relation: str,
) -> None:
    key = (from_id, to_id, edge_type, relation)
    if key in edge_set:
        return
    edge_set.add(key)
    edges.append(Edge(from_id=from_id, to_id=to_id, type=edge_type, relation=relation))


def build_rpg(
    repo_root: Path,
    out_dir: Path,
    max_file_bytes: int = MAX_FILE_BYTES_DEFAULT,
    progress_interval: int = 100,
) -> dict[str, object]:
    semantic_cache_path = out_dir / "semantic_cache.json"
    semantic_cache = read_semantic_cache(semantic_cache_path)
    next_cache: dict[str, object] = {}

    source_files = collect_source_files(repo_root, exclude_dirs=DEFAULT_EXCLUDE_DIRS)
    if not source_files:
        raise RuntimeError("No source files found. Check repository root and filters.")

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    edge_set: set[tuple[str, str, str, str]] = set()

    file_to_includes: dict[str, list[str]] = defaultdict(list)
    symbol_body_text: dict[str, str] = {}
    function_name_to_ids: dict[str, list[str]] = defaultdict(list)

    root_id = "dir:."
    nodes[root_id] = Node(id=root_id, kind="directory", feature=["organize repository root"], path=".")

    rel_files: list[str] = []
    language_by_file: dict[str, str] = {}
    for source_file in source_files:
        language = language_for_file(source_file)
        if language is None:
            continue
        rel = to_posix_rel(source_file, repo_root)
        rel_files.append(rel)
        language_by_file[rel] = language

    all_file_set = set(rel_files)
    basename_to_paths: dict[str, list[str]] = defaultdict(list)
    for rel in rel_files:
        basename_to_paths[PurePosixPath(rel).name].append(rel)

    total_files = len(rel_files)
    skipped_large_files = 0

    for idx, rel in enumerate(rel_files, start=1):
        if progress_interval > 0 and (idx % progress_interval == 0 or idx == total_files):
            print(f"[rpg] indexed {idx}/{total_files} files", file=sys.stderr, flush=True)

        path = repo_root / rel
        language = language_by_file[rel]
        file_size = path.stat().st_size
        text = ""
        parse_text = file_size <= max_file_bytes
        if parse_text:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            skipped_large_files += 1

        rel_parent = str(PurePosixPath(rel).parent)
        dir_chain: list[str] = []
        cur = PurePosixPath(rel_parent)
        while str(cur) not in ("", "."):
            dir_chain.append(str(cur))
            cur = cur.parent
        dir_chain.reverse()
        for directory in dir_chain:
            dir_id = f"dir:{directory}"
            if dir_id not in nodes:
                nodes[dir_id] = Node(
                    id=dir_id,
                    kind="directory",
                    feature=make_directory_features(directory),
                    path=directory,
                )
            parent = str(PurePosixPath(directory).parent)
            parent_id = root_id if parent in ("", ".") else f"dir:{parent}"
            if parent_id not in nodes:
                parent_path = "." if parent in ("", ".") else parent
                nodes[parent_id] = Node(
                    id=parent_id,
                    kind="directory",
                    feature=make_directory_features(parent_path),
                    path=parent_path,
                )
            add_edge(edge_set, edges, parent_id, dir_id, "functional", "contains")

        file_id = f"file:{rel}"
        line_count = (text.count("\n") + 1) if parse_text and text else None
        nodes[file_id] = Node(
            id=file_id,
            kind="file",
            feature=make_file_features(rel),
            path=rel,
            language=language,
            start_line=1 if line_count is not None else None,
            end_line=line_count,
        )
        parent_id = root_id if rel_parent in ("", ".") else f"dir:{rel_parent}"
        add_edge(edge_set, edges, parent_id, file_id, "functional", "contains")

        if parse_text:
            symbols = parse_symbols_for_file(language, text, rel)
            for symbol in symbols:
                node_prefix = "fn" if symbol.kind == "function" else "class"
                node_id = f"{node_prefix}:{rel}:{symbol.name}:{symbol.start_line}"
                features = make_symbol_features(
                    symbol.name,
                    rel,
                    symbol.kind,
                    semantic_cache=semantic_cache,
                    signature_hash=symbol.signature_hash,
                )
                cache_key = f"{rel}:{symbol.kind}:{symbol.name}"
                next_cache[cache_key] = {"hash": symbol.signature_hash, "features": features}
                nodes[node_id] = Node(
                    id=node_id,
                    kind=symbol.kind,
                    feature=features,
                    path=rel,
                    symbol=symbol.name,
                    language=language,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                )
                add_edge(edge_set, edges, file_id, node_id, "functional", "contains")
                if symbol.kind == "function":
                    simple_name = symbol.name.split("::")[-1]
                    function_name_to_ids[simple_name].append(node_id)
                if symbol.kind == "function" and symbol.body_text:
                    symbol_body_text[node_id] = symbol.body_text

            if language in ("cpp", "c"):
                includes = parse_cpp_includes(text)
                if includes:
                    file_to_includes[rel].extend(includes)

    categories: set[str] = {infer_category(rel) for rel in rel_files}
    categories.add(DEFAULT_CATEGORY)
    for category in sorted(categories):
        category_slug = slugify(category)
        category_path = f"<functional>/{category_slug}"
        category_id = f"dir:{category_path}"
        nodes[category_id] = Node(
            id=category_id,
            kind="directory",
            feature=[f"organize {category.lower()} scope"],
            path=category_path,
            symbol=category,
        )
        add_edge(edge_set, edges, root_id, category_id, "functional", "contains")

    for rel in rel_files:
        category_id = f"dir:<functional>/{slugify(infer_category(rel))}"
        add_edge(edge_set, edges, category_id, f"file:{rel}", "functional", "scopes")

    for rel, includes in file_to_includes.items():
        from_id = f"file:{rel}"
        for include_target in includes:
            resolved = resolve_include_path(
                current_file=rel,
                include_target=include_target,
                all_files=all_file_set,
                basename_to_paths=basename_to_paths,
            )
            if resolved is None:
                continue
            to_id = f"file:{resolved}"
            if to_id in nodes:
                add_edge(edge_set, edges, from_id, to_id, "dependency", "includes")

    for source_id, body in symbol_body_text.items():
        source_path = nodes[source_id].path
        call_names = {m.group(1) for m in CALL_TOKEN_RE.finditer(body)}
        call_names = {name for name in call_names if name not in FUNCTION_KEYWORDS}
        for call_name in call_names:
            candidates = function_name_to_ids.get(call_name, [])
            if not candidates:
                continue
            local = [c for c in candidates if nodes[c].path == source_path]
            targets = local if local else candidates
            if len(targets) > 3:
                continue
            for target in targets:
                if target != source_id:
                    add_edge(edge_set, edges, source_id, target, "dependency", "calls")

    write_semantic_cache(semantic_cache_path, next_cache)
    return {
        "nodes": nodes,
        "edges": edges,
        "rel_files": rel_files,
        "skipped_large_files": skipped_large_files,
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rpg_json(rpg: dict[str, object], repo_root: Path, json_path: Path) -> None:
    ensure_parent_dir(json_path)
    nodes = list(rpg["nodes"].values())  # type: ignore[index]
    edges = rpg["edges"]  # type: ignore[index]
    payload = {
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "file_count": len(rpg["rel_files"]),  # type: ignore[index]
        },
        "nodes": [node.to_json() for node in nodes],
        "edges": [edge.to_json() for edge in edges],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_fts_text(node: Node) -> str:
    fields = [node.id, node.kind]
    if node.path:
        fields.append(node.path.replace("/", " "))
    if node.symbol:
        fields.append(node.symbol)
    fields.extend(node.feature)
    return " ".join(fields)


def write_rpg_sqlite(rpg: dict[str, object], db_path: Path) -> bool:
    ensure_parent_dir(db_path)
    conn = sqlite3.connect(str(db_path))
    # Some sandboxed Windows filesystems reject SQLite journaling/locking.
    # This index is rebuildable, so prefer maximum compatibility over durability.
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("DROP TABLE IF EXISTS nodes;")
    conn.execute("DROP TABLE IF EXISTS edges;")
    conn.execute("DROP TABLE IF EXISTS nodes_search;")
    conn.execute("DROP TABLE IF EXISTS nodes_fts;")
    conn.execute(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            feature_json TEXT NOT NULL,
            path TEXT,
            symbol TEXT,
            language TEXT,
            start_line INTEGER,
            end_line INTEGER
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            type TEXT NOT NULL,
            relation TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX edges_from_idx ON edges(from_id);")
    conn.execute("CREATE INDEX edges_to_idx ON edges(to_id);")
    conn.execute("CREATE INDEX edges_type_idx ON edges(type);")

    fts_enabled = True
    try:
        conn.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(id UNINDEXED, text);")
    except sqlite3.OperationalError:
        fts_enabled = False
        conn.execute("CREATE TABLE nodes_search (id TEXT PRIMARY KEY, text TEXT NOT NULL);")
        conn.execute("CREATE INDEX nodes_search_text_idx ON nodes_search(text);")

    node_rows = []
    search_rows = []
    for node in rpg["nodes"].values():  # type: ignore[index]
        node_rows.append(
            (
                node.id,
                node.kind,
                json.dumps(node.feature, ensure_ascii=True),
                node.path,
                node.symbol,
                node.language,
                node.start_line,
                node.end_line,
            )
        )
        search_rows.append((node.id, build_fts_text(node)))
    conn.executemany(
        """
        INSERT INTO nodes (id, kind, feature_json, path, symbol, language, start_line, end_line)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        node_rows,
    )
    conn.executemany(
        "INSERT INTO edges (from_id, to_id, type, relation) VALUES (?, ?, ?, ?);",
        [(e.from_id, e.to_id, e.type, e.relation) for e in rpg["edges"]],  # type: ignore[index]
    )
    if fts_enabled:
        conn.executemany("INSERT INTO nodes_fts (id, text) VALUES (?, ?);", search_rows)
    else:
        conn.executemany("INSERT INTO nodes_search (id, text) VALUES (?, ?);", search_rows)
    conn.commit()
    conn.close()
    return fts_enabled


def open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"RPG database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def fts_available(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts';").fetchone()
    return row is not None


def query_search(conn: sqlite3.Connection, intent: str, limit: int) -> list[dict[str, object]]:
    terms = [t for t in normalize_text_words(intent) if t]
    if not terms:
        return []
    if fts_available(conn):
        expr = " OR ".join(terms)
        rows = conn.execute(
            """
            SELECT n.id, n.kind, n.path, n.symbol, n.language, n.start_line, n.end_line, n.feature_json,
                   bm25(nodes_fts) AS score
            FROM nodes_fts
            JOIN nodes AS n ON n.id = nodes_fts.id
            WHERE nodes_fts MATCH ?
            ORDER BY score
            LIMIT ?;
            """,
            (expr, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "score": float(row["score"]),
                "meta": {
                    "path": row["path"],
                    "symbol": row["symbol"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                },
                "feature": json.loads(row["feature_json"]),
            }
            for row in rows
        ]
    like_pattern = f"%{' %'.join(terms)}%"
    rows = conn.execute(
        """
        SELECT n.id, n.kind, n.path, n.symbol, n.language, n.start_line, n.end_line, n.feature_json
        FROM nodes_search AS s
        JOIN nodes AS n ON n.id = s.id
        WHERE s.text LIKE ?
        LIMIT ?;
        """,
        (like_pattern, limit),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "score": 0.0,
            "meta": {
                "path": row["path"],
                "symbol": row["symbol"],
                "language": row["language"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
            },
            "feature": json.loads(row["feature_json"]),
        }
        for row in rows
    ]


def fetch_node(conn: sqlite3.Connection, node_id: str) -> dict[str, object]:
    node = conn.execute(
        """
        SELECT id, kind, feature_json, path, symbol, language, start_line, end_line
        FROM nodes
        WHERE id = ?;
        """,
        (node_id,),
    ).fetchone()
    if node is None:
        raise KeyError(f"Node not found: {node_id}")
    outgoing = conn.execute(
        """
        SELECT from_id, to_id, type, relation
        FROM edges
        WHERE from_id = ?
        ORDER BY type, relation, to_id;
        """,
        (node_id,),
    ).fetchall()
    incoming = conn.execute(
        """
        SELECT from_id, to_id, type, relation
        FROM edges
        WHERE to_id = ?
        ORDER BY type, relation, from_id;
        """,
        (node_id,),
    ).fetchall()
    return {
        "node": {
            "id": node["id"],
            "kind": node["kind"],
            "feature": json.loads(node["feature_json"]),
            "meta": {
                "path": node["path"],
                "symbol": node["symbol"],
                "language": node["language"],
                "start_line": node["start_line"],
                "end_line": node["end_line"],
            },
        },
        "outgoing": [dict(row) for row in outgoing],
        "incoming": [dict(row) for row in incoming],
    }


def explore_graph(
    conn: sqlite3.Connection,
    node_id: str,
    edge_type: str | None,
    depth: int,
    direction: str,
) -> dict[str, object]:
    if depth < 0:
        raise ValueError("depth must be >= 0")
    params: tuple[object, ...] = ()
    edge_where = ""
    if edge_type and edge_type != "any":
        edge_where = "WHERE type = ?"
        params = (edge_type,)
    rows = conn.execute(f"SELECT from_id, to_id, type, relation FROM edges {edge_where};", params).fetchall()

    out_adj: dict[str, list[dict[str, str]]] = defaultdict(list)
    in_adj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        edge = {"from": row["from_id"], "to": row["to_id"], "type": row["type"], "relation": row["relation"]}
        out_adj[row["from_id"]].append(edge)
        in_adj[row["to_id"]].append(edge)

    visited: set[str] = {node_id}
    traversed: set[tuple[str, str, str, str]] = set()
    q: deque[tuple[str, int]] = deque([(node_id, 0)])
    while q:
        cur, d = q.popleft()
        if d >= depth:
            continue
        candidates: list[dict[str, str]] = []
        if direction in ("out", "both"):
            candidates.extend(out_adj.get(cur, []))
        if direction in ("in", "both"):
            candidates.extend(in_adj.get(cur, []))
        for edge in candidates:
            nxt = edge["to"] if edge["from"] == cur else edge["from"]
            traversed.add((edge["from"], edge["to"], edge["type"], edge["relation"]))
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, d + 1))

    placeholders = ",".join("?" for _ in visited)
    node_rows = conn.execute(
        f"""
        SELECT id, kind, feature_json, path, symbol, language, start_line, end_line
        FROM nodes
        WHERE id IN ({placeholders});
        """,
        tuple(visited),
    ).fetchall()
    return {
        "root": node_id,
        "edge_type": edge_type or "any",
        "depth": depth,
        "direction": direction,
        "nodes": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "feature": json.loads(row["feature_json"]),
                "meta": {
                    "path": row["path"],
                    "symbol": row["symbol"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                },
            }
            for row in node_rows
        ],
        "edges": [
            {"from": a, "to": b, "type": c, "relation": d}
            for a, b, c, d in sorted(traversed)
        ],
    }


def as_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def cmd_build(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rpg = build_rpg(
        repo_root=repo_root,
        out_dir=out_dir,
        max_file_bytes=args.max_file_bytes,
        progress_interval=args.progress_interval,
    )
    json_path = out_dir / "rpg.json"
    db_path = out_dir / "rpg.sqlite"
    write_rpg_json(rpg=rpg, repo_root=repo_root, json_path=json_path)
    fts_enabled = write_rpg_sqlite(rpg=rpg, db_path=db_path)
    print(
        as_json(
            {
                "ok": True,
                "repo_root": str(repo_root),
                "out_dir": str(out_dir),
                "json_path": str(json_path),
                "sqlite_path": str(db_path),
                "fts_enabled": fts_enabled,
                "node_count": len(rpg["nodes"]),
                "edge_count": len(rpg["edges"]),
                "file_count": len(rpg["rel_files"]),
                "skipped_large_files": rpg["skipped_large_files"],
                "max_file_bytes": args.max_file_bytes,
            }
        )
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = open_db(Path(args.db).resolve())
    try:
        results = query_search(conn, intent=args.intent, limit=args.limit)
        print(as_json({"intent": args.intent, "results": results}))
    finally:
        conn.close()
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    conn = open_db(Path(args.db).resolve())
    try:
        print(as_json(fetch_node(conn, node_id=args.node_id)))
    finally:
        conn.close()
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    conn = open_db(Path(args.db).resolve())
    try:
        result = explore_graph(
            conn,
            node_id=args.node_id,
            edge_type=args.edge_type,
            depth=args.depth,
            direction=args.direction,
        )
        print(as_json(result))
    finally:
        conn.close()
    return 0


class RPGRequestHandler(BaseHTTPRequestHandler):
    db_path: Path

    def _json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_error(self, status: int, message: str) -> None:
        self._json_response(status, {"ok": False, "error": message})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._json_response(200, {"ok": True})
            return
        try:
            conn = open_db(self.db_path)
        except Exception as exc:  # pylint: disable=broad-except
            self._handle_error(500, f"failed to open DB: {exc}")
            return
        try:
            if parsed.path == "/search":
                intent = (query.get("intent") or [""])[0].strip()
                limit_raw = (query.get("limit") or ["10"])[0]
                if not intent:
                    self._handle_error(400, "missing query parameter: intent")
                    return
                try:
                    limit = max(1, min(200, int(limit_raw)))
                except ValueError:
                    self._handle_error(400, "limit must be an integer")
                    return
                results = query_search(conn, intent=intent, limit=limit)
                self._json_response(200, {"ok": True, "intent": intent, "results": results})
                return
            if parsed.path == "/fetch":
                node_id = (query.get("node_id") or [""])[0].strip()
                if not node_id:
                    self._handle_error(400, "missing query parameter: node_id")
                    return
                self._json_response(200, {"ok": True, **fetch_node(conn, node_id=node_id)})
                return
            if parsed.path == "/explore":
                node_id = (query.get("node_id") or [""])[0].strip()
                edge_type = (query.get("edge_type") or ["any"])[0]
                direction = (query.get("direction") or ["both"])[0]
                depth_raw = (query.get("depth") or ["2"])[0]
                if not node_id:
                    self._handle_error(400, "missing query parameter: node_id")
                    return
                if direction not in {"in", "out", "both"}:
                    self._handle_error(400, "direction must be in|out|both")
                    return
                try:
                    depth = max(0, min(6, int(depth_raw)))
                except ValueError:
                    self._handle_error(400, "depth must be an integer")
                    return
                result = explore_graph(
                    conn,
                    node_id=node_id,
                    edge_type=edge_type,
                    depth=depth,
                    direction=direction,
                )
                self._json_response(200, {"ok": True, **result})
                return
            self._handle_error(404, f"unknown endpoint: {parsed.path}")
        except KeyError as exc:
            self._handle_error(404, str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            self._handle_error(500, str(exc))
        finally:
            conn.close()

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def cmd_serve(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"RPG database not found: {db_path}")
    handler_type = type("BoundRPGRequestHandler", (RPGRequestHandler,), {"db_path": db_path})
    server = ThreadingHTTPServer((args.host, args.port), handler_type)
    print(
        as_json(
            {
                "ok": True,
                "host": args.host,
                "port": args.port,
                "db": str(db_path),
                "endpoints": ["/health", "/search", "/fetch", "/explore"],
            }
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repository Planning Graph CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build local RPG JSON + SQLite")
    p_build.add_argument("--repo", default=".", help="Repository root to index")
    p_build.add_argument("--out", default=".rpg_data", help="Output directory for RPG artifacts")
    p_build.add_argument(
        "--max-file-bytes",
        type=int,
        default=MAX_FILE_BYTES_DEFAULT,
        help=f"Skip parsing symbols/includes for files larger than this size (default: {MAX_FILE_BYTES_DEFAULT})",
    )
    p_build.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print build progress every N files (default: 100, set 0 to disable)",
    )
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="Search nodes by intent")
    p_search.add_argument("intent", help="Natural-language intent")
    p_search.add_argument("--db", default=".rpg_data/rpg.sqlite", help="Path to RPG SQLite DB")
    p_search.add_argument("--limit", type=int, default=10, help="Max number of matches")
    p_search.set_defaults(func=cmd_search)

    p_fetch = sub.add_parser("fetch", help="Fetch one node with edges")
    p_fetch.add_argument("node_id", help="Node ID")
    p_fetch.add_argument("--db", default=".rpg_data/rpg.sqlite", help="Path to RPG SQLite DB")
    p_fetch.set_defaults(func=cmd_fetch)

    p_explore = sub.add_parser("explore", help="Traverse RPG from a root node")
    p_explore.add_argument("node_id", help="Root node ID")
    p_explore.add_argument("--db", default=".rpg_data/rpg.sqlite", help="Path to RPG SQLite DB")
    p_explore.add_argument("--edge-type", default="any", help="functional, dependency, or any")
    p_explore.add_argument("--depth", type=int, default=2, help="Traversal depth")
    p_explore.add_argument("--direction", choices=["in", "out", "both"], default="both")
    p_explore.set_defaults(func=cmd_explore)

    p_serve = sub.add_parser("serve", help="Serve RPG query endpoints over HTTP")
    p_serve.add_argument("--db", default=".rpg_data/rpg.sqlite", help="Path to RPG SQLite DB")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve.add_argument("--port", type=int, default=7878, help="Bind port")
    p_serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str]) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pylint: disable=broad-except
        print(as_json({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
