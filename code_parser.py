"""Parse a codebase into semantic chunks (functions, methods, classes,
interfaces) using tree-sitter, across multiple languages.

Design notes
------------
Rather than per-language regex parsing, we use tree-sitter grammars via
`tree-sitter-language-pack` so the same traversal logic works across
languages, with a small per-language config describing which node types
count as "definitions" and how to pull out names/bodies/docstrings.

Each extracted CodeChunk gets a stable id of "{file_path}::{qualified_name}"
so that a later run (after code changes) can be diffed against a prior
index by id, and a content_hash so we can tell whether a chunk that kept
its name actually changed in a way docs might care about.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from models import CodeChunk

# ---------------------------------------------------------------------------
# Per-language configuration
# ---------------------------------------------------------------------------


@dataclass
class LangConfig:
    ts_name: str  # name passed to tree_sitter_language_pack.get_parser
    extensions: tuple[str, ...]
    class_types: tuple[str, ...]  # node types that open a new "class" scope
    function_types: tuple[str, ...]  # node types that are function/method defs
    name_field: str  # field name holding the identifier node
    body_field: str  # field name holding the body/block node
    docstring_fn: Callable[[Node, bytes], Optional[str]]


def _py_docstring(def_node: Node, src: bytes) -> Optional[str]:
    body = def_node.child_by_field_name("body")
    if not body or body.child_count == 0:
        return None
    first = body.children[0]
    string_node = None
    if first.type == "string":
        string_node = first
    elif first.type == "expression_statement" and first.child_count and first.children[0].type == "string":
        string_node = first.children[0]
    if string_node is None:
        return None
    # Prefer the inner string_content node so we don't have to manually
    # strip quote characters/prefixes (r"...", f"...", etc).
    content_nodes = [c for c in string_node.children if c.type == "string_content"]
    if content_nodes:
        text = "\n".join(c.text.decode("utf-8", "replace") for c in content_nodes)
    else:
        text = string_node.text.decode("utf-8", "replace").strip("\"'rRbBfF")
    return text.strip() or None


_COMMENT_NODE_TYPES = ("comment", "block_comment", "line_comment")


def _leading_comment_docstring(prefix: str) -> Callable[[Node, bytes], Optional[str]]:
    """For JSDoc/Javadoc-style: a /** ... */ comment immediately preceding
    the definition node. Handles both grammars where annotations/decorators
    sit as siblings before the def (skip over them) and grammars (e.g. Java)
    where they're nested inside the def's own `modifiers` child (no skip
    needed, since the comment is already the direct prev_sibling)."""

    def _fn(def_node: Node, src: bytes) -> Optional[str]:
        prev = def_node.prev_sibling
        while prev is not None and prev.type in ("decorator", "marker_annotation", "annotation", "modifiers"):
            prev = prev.prev_sibling
        if prev is not None and prev.type in _COMMENT_NODE_TYPES:
            text = prev.text.decode("utf-8", "replace")
            if text.strip().startswith(prefix):
                raw_lines = [ln.strip() for ln in text.splitlines()]
                raw_lines = [ln for ln in raw_lines if ln not in ("/**", "*/", "**/")]
                cleaned = [ln.lstrip("*").strip() for ln in raw_lines]
                cleaned = [ln for ln in cleaned if ln]
                return "\n".join(cleaned).strip() or None
        return None

    return _fn


def _go_docstring(def_node: Node, src: bytes) -> Optional[str]:
    prev = def_node.prev_sibling
    lines = []
    while prev is not None and prev.type == "comment":
        text = prev.text.decode("utf-8", "replace").lstrip("/").strip()
        lines.insert(0, text)
        prev = prev.prev_sibling
    return "\n".join(lines).strip() or None


LANGUAGES: dict[str, LangConfig] = {
    "python": LangConfig(
        ts_name="python",
        extensions=(".py",),
        class_types=("class_definition",),
        function_types=("function_definition",),
        name_field="name",
        body_field="body",
        docstring_fn=_py_docstring,
    ),
    "javascript": LangConfig(
        ts_name="javascript",
        extensions=(".js", ".jsx", ".mjs"),
        class_types=("class_declaration",),
        function_types=("function_declaration", "method_definition"),
        name_field="name",
        body_field="body",
        docstring_fn=_leading_comment_docstring("/**"),
    ),
    "typescript": LangConfig(
        ts_name="typescript",
        extensions=(".ts",),
        class_types=("class_declaration", "interface_declaration"),
        function_types=("function_declaration", "method_definition", "method_signature"),
        name_field="name",
        body_field="body",
        docstring_fn=_leading_comment_docstring("/**"),
    ),
    "tsx": LangConfig(
        ts_name="tsx",
        extensions=(".tsx",),
        class_types=("class_declaration", "interface_declaration"),
        function_types=("function_declaration", "method_definition", "method_signature"),
        name_field="name",
        body_field="body",
        docstring_fn=_leading_comment_docstring("/**"),
    ),
    "java": LangConfig(
        ts_name="java",
        extensions=(".java",),
        class_types=("class_declaration", "interface_declaration"),
        function_types=("method_declaration", "constructor_declaration"),
        name_field="name",
        body_field="body",
        docstring_fn=_leading_comment_docstring("/**"),
    ),
    "go": LangConfig(
        ts_name="go",
        extensions=(".go",),
        class_types=("type_declaration",),
        function_types=("function_declaration", "method_declaration"),
        name_field="name",
        body_field="body",
        docstring_fn=_go_docstring,
    ),
}

EXT_TO_LANG = {ext: lang for lang, cfg in LANGUAGES.items() for ext in cfg.extensions}

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "target", ".mypy_cache", ".pytest_cache", "vendor",
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


_LINE_COMMENT_RE = re.compile(r"(#|//).*?$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_diff(text: str) -> str:
    """Best-effort language-agnostic normalization used only to tell
    cosmetic edits (reformatting, comment tweaks) apart from real changes.
    Strips '#'/'//' line comments and '/* */' block comments (imprecise for
    strings containing these sequences, acceptable for a change-priority
    heuristic) and collapses all whitespace."""
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _signature(node: Node, cfg: LangConfig, src: bytes) -> str:
    """Everything from the node's start up to (not including) the body
    block, i.e. the declaration line(s) without the implementation.

    Some grammars (e.g. Python) attach a comment that sits between the ':'
    and the first real statement as a sibling *before* the body field
    rather than as the body's first child -- walk backward over any such
    leading comments so they don't get counted as part of the signature."""
    body = node.child_by_field_name(cfg.body_field)
    if body is None:
        end_byte = node.end_byte
    else:
        children = node.children
        try:
            idx = children.index(body)
        except ValueError:
            idx = None
        end_byte = body.start_byte
        if idx is not None:
            j = idx - 1
            while j >= 0 and children[j].type in ("comment", "line_comment", "block_comment"):
                end_byte = children[j].start_byte
                j -= 1
    text = src[node.start_byte:end_byte].decode("utf-8", "replace")
    return text.strip().rstrip(":").strip()


def _node_name(node: Node, cfg: LangConfig) -> Optional[str]:
    name_node = node.child_by_field_name(cfg.name_field)
    if name_node is None:
        return None
    return name_node.text.decode("utf-8", "replace")


def _body_text(node: Node, cfg: LangConfig, src: bytes) -> str:
    body = node.child_by_field_name(cfg.body_field)
    if body is None:
        return ""
    return src[body.start_byte:body.end_byte].decode("utf-8", "replace")


def _make_chunk(node: Node, cfg: LangConfig, src: bytes, lang: str, rel_path: str,
                 qualified: str, name: str, kind: str) -> CodeChunk:
    signature = _signature(node, cfg, src)
    docstring = cfg.docstring_fn(node, src)
    body_text = _body_text(node, cfg, src)
    full_text = node.text.decode("utf-8", "replace")
    return CodeChunk(
        id=f"{rel_path}::{qualified}",
        file_path=rel_path,
        language=lang,
        kind=kind,
        name=name,
        qualified_name=qualified,
        signature=signature,
        docstring=docstring,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        source_text=full_text,
        content_hash=_hash(full_text),
        signature_hash=_hash(signature),
        docstring_hash=_hash(docstring or ""),
        body_hash=_hash(body_text),
        body_hash_normalized=_hash(_normalize_for_diff(body_text)),
    )


def parse_source(src: bytes, rel_path: str, language: Optional[str] = None) -> list[CodeChunk]:
    """Parse raw source bytes (not necessarily on disk -- e.g. content read
    via `git show <ref>:<path>`) into CodeChunks. `language` can be passed
    explicitly when it can't be inferred from `rel_path`'s extension."""
    lang = language or EXT_TO_LANG.get(os.path.splitext(rel_path)[1])
    if lang is None:
        return []
    cfg = LANGUAGES[lang]
    parser = get_parser(cfg.ts_name)
    tree = parser.parse(src)

    chunks: list[CodeChunk] = []

    def walk(node: Node, class_stack: list[str]):
        for child in node.children:
            if child.type in cfg.class_types:
                cname = _node_name(child, cfg) or "<anonymous>"
                qualified = ".".join(class_stack + [cname])
                body_node = child.child_by_field_name(cfg.body_field)
                chunks.append(_make_chunk(
                    child, cfg, src, lang, rel_path, qualified, cname,
                    kind="interface" if "interface" in child.type else "class",
                ))
                walk(body_node if body_node else child, class_stack + [cname])
            elif child.type in cfg.function_types:
                fname = _node_name(child, cfg) or "<anonymous>"
                qualified = ".".join(class_stack + [fname])
                chunks.append(_make_chunk(
                    child, cfg, src, lang, rel_path, qualified, fname,
                    kind="method" if class_stack else "function",
                ))
                # don't recurse into function bodies for nested defs in v1
            else:
                walk(child, class_stack)

    walk(tree.root_node, [])
    return chunks


def parse_file(file_path: str, repo_root: str) -> list[CodeChunk]:
    ext = os.path.splitext(file_path)[1]
    if ext not in EXT_TO_LANG:
        return []
    with open(file_path, "rb") as f:
        src = f.read()
    rel_path = os.path.relpath(file_path, repo_root)
    return parse_source(src, rel_path)


def parse_codebase(repo_root: str, ignore_dirs: Optional[set[str]] = None) -> list[CodeChunk]:
    ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
    all_chunks: list[CodeChunk] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                all_chunks.extend(parse_file(fpath, repo_root))
            except Exception as e:  # noqa: BLE001 - keep indexing resilient to one bad file
                print(f"[code_parser] skipping {fpath}: {e}")
    return all_chunks


if __name__ == "__main__":
    import sys
    import json

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    result = parse_codebase(root)
    print(json.dumps([c.to_dict() for c in result], indent=2))
    print(f"\n# {len(result)} chunks extracted", file=sys.stderr)
