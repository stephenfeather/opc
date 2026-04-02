"""Knowledge graph entity and relationship extraction from learning content.

Extracts entities (files, modules, tools, concepts, errors, config vars, languages,
libraries) and infers relationships between them using heuristic rules. Persists
results to PostgreSQL kg_entities, kg_edges, and kg_entity_mentions tables.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    name: str  # canonical (lowercase, normalized)
    display_name: str  # original casing
    entity_type: str
    span: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    source: str  # entity canonical name
    target: str  # entity canonical name
    relation: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Known dictionaries
# ---------------------------------------------------------------------------

KNOWN_TOOLS: set[str] = {
    "pytest", "ruff", "black", "mypy", "pyright", "docker", "git", "npm", "uv",
    "pip", "poetry", "cargo", "rustc", "go", "node", "deno", "bun", "webpack",
    "vite", "eslint", "prettier", "terraform", "ansible", "kubectl", "helm",
    "redis", "nginx", "gunicorn", "uvicorn", "celery", "flask", "django",
    "fastapi", "express", "nextjs", "react", "vue", "svelte", "tailwind",
    "postgres", "postgresql", "mysql", "sqlite", "mongodb", "neo4j",
    "xcodebuild", "swift", "swiftc", "clang", "gcc", "make", "cmake",
    "bat", "eza", "fd", "rg", "ripgrep", "jq", "curl", "wget",
}

KNOWN_LANGUAGES: set[str] = {
    "python", "typescript", "javascript", "rust", "go", "swift", "java",
    "kotlin", "c", "cpp", "c++", "sql", "bash", "zsh", "shell", "ruby",
    "php", "lua", "zig", "haskell", "lean", "lean4", "html", "css",
}

KNOWN_LIBRARIES: set[str] = {
    "pgvector", "asyncpg", "psycopg2", "sqlalchemy", "aiohttp", "httpx",
    "requests", "pydantic", "numpy", "pandas", "scikit-learn", "scipy",
    "torch", "pytorch", "tensorflow", "transformers", "openai", "anthropic",
    "langchain", "chromadb", "pinecone", "weaviate", "hdbscan", "spacy",
    "nltk", "beautifulsoup", "scrapy", "celery", "boto3", "paramiko",
    "click", "typer", "rich", "textual", "pytest", "unittest", "hypothesis",
}

# File extensions that indicate a real file path (not a version like 3.12)
_FILE_EXTENSIONS: set[str] = {
    "py", "ts", "tsx", "js", "jsx", "rs", "go", "swift", "java", "kt",
    "c", "h", "cpp", "hpp", "sql", "sh", "bash", "zsh", "yaml", "yml",
    "json", "toml", "cfg", "ini", "md", "txt", "html", "css", "scss",
    "vue", "svelte", "rb", "php", "lua", "zig", "lean", "ex", "exs",
    "env", "lock", "log", "csv", "xml", "graphql", "proto", "dockerfile",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# File paths: must contain / or end with a known extension
_RE_FILE_PATH = re.compile(
    r'(?:^|[\s`"\',;()\[\]{}])('
    r'(?:[\w.~-]+/)+[\w.-]+'  # path with at least one /
    r'|'
    r'[\w.-]+\.(?:' + "|".join(_FILE_EXTENSIONS) + r')'  # or file.ext
    r')(?=[\s`"\',;()\[\]{}]|$)',
    re.MULTILINE,
)

# Python imports: from X import Y or import X
_RE_PYTHON_IMPORT = re.compile(
    r'(?:from|import)\s+([\w.]+)',
)

# Environment variables: ALL_CAPS_WITH_UNDERSCORES (min 4 chars, min 1 underscore)
_RE_ENV_VAR = re.compile(
    r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b',
)

# Error types: SomethingError, SomethingException, SomethingWarning
_RE_ERROR_TYPE = re.compile(
    r'\b(\w+(?:Error|Exception|Warning))\b',
)

# Quoted terms: backtick, single, or double quoted (2-40 chars)
_RE_QUOTED = re.compile(
    r'[`]([^`]{2,40})[`]',
)

# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _normalize_name(name: str, entity_type: str) -> str:
    """Produce a canonical lowercase name."""
    n = name.strip().lower()
    if entity_type == "file":
        # Remove leading ./ or /
        n = n.lstrip("./")
    return n


def _is_noise(name: str) -> bool:
    """Filter out common false-positive entities."""
    # Too short
    if len(name) < 2:
        return True
    # Pure numbers or versions
    if re.match(r'^[\d.]+$', name):
        return True
    # Common English words that match patterns
    noise_words = {
        "the", "and", "for", "with", "this", "that", "from", "have",
        "not", "are", "was", "were", "been", "will", "would", "could",
        "should", "can", "may", "might", "shall", "must", "need",
        "true", "false", "none", "null", "undefined",
    }
    if name.lower() in noise_words:
        return True
    return False


def extract_entities(content: str) -> list[ExtractedEntity]:
    """Extract entities from learning content using heuristic rules.

    Applies extractors in priority order. Deduplicates by (name, type).
    """
    seen: dict[tuple[str, str], ExtractedEntity] = {}

    def _add(display: str, etype: str, span: tuple[int, int] | None = None,
             meta: dict | None = None):
        canon = _normalize_name(display, etype)
        if _is_noise(canon):
            return
        key = (canon, etype)
        if key not in seen:
            seen[key] = ExtractedEntity(
                name=canon,
                display_name=display.strip(),
                entity_type=etype,
                span=span,
                metadata=meta or {},
            )

    # 1. File paths
    for m in _RE_FILE_PATH.finditer(content):
        path = m.group(1).rstrip(".,;:!?)")
        # Validate: must have / or a known extension
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if "/" in path or ext in _FILE_EXTENSIONS:
            _add(path, "file", (m.start(1), m.end(1)))

    # 2. Python imports
    for m in _RE_PYTHON_IMPORT.finditer(content):
        module = m.group(1)
        _add(module, "module", (m.start(1), m.end(1)))

    # 3. Environment variables
    for m in _RE_ENV_VAR.finditer(content):
        var = m.group(1)
        # Skip if it looks like a constant in code context (< 4 chars after split)
        parts = var.split("_")
        if len(parts) >= 2 and all(len(p) >= 1 for p in parts):
            _add(var, "config", (m.start(1), m.end(1)))

    # 4. Error types
    for m in _RE_ERROR_TYPE.finditer(content):
        err = m.group(1)
        _add(err, "error", (m.start(1), m.end(1)))

    # 5. Known tools (word-boundary match)
    for tool in KNOWN_TOOLS:
        # Use word boundary search
        pattern = re.compile(r'\b' + re.escape(tool) + r'\b', re.IGNORECASE)
        m = pattern.search(content)
        if m:
            _add(m.group(0), "tool", (m.start(), m.end()))

    # 6. Known languages
    for lang in KNOWN_LANGUAGES:
        pattern = re.compile(r'\b' + re.escape(lang) + r'\b', re.IGNORECASE)
        m = pattern.search(content)
        if m:
            _add(m.group(0), "language", (m.start(), m.end()))

    # 7. Known libraries
    for lib in KNOWN_LIBRARIES:
        pattern = re.compile(r'\b' + re.escape(lib) + r'\b', re.IGNORECASE)
        m = pattern.search(content)
        if m:
            _add(m.group(0), "library", (m.start(), m.end()))

    # 8. Backtick-quoted terms (likely technical terms)
    for m in _RE_QUOTED.finditer(content):
        term = m.group(1).strip()
        # Skip if already captured as another type
        canon = _normalize_name(term, "concept")
        already = any(canon == k[0] for k in seen)
        if not already and not _is_noise(canon):
            # Classify: if it looks like a file, tool, etc., skip
            has_dot = "." in term.split()[-1] if " " not in term else False
            if "/" in term or has_dot:
                continue
            _add(term, "concept", (m.start(1), m.end(1)))

    return list(seen.values())


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

# Sentence splitter: split on period followed by space+uppercase, or !/?/newline
# Avoids splitting on dots inside filenames like reranker.py
_RE_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|\n+')

# Relation signal patterns
_RE_SUPERSEDES = re.compile(
    r'\b(?:instead of|replaced?|supersede[sd]?|deprecate[sd]?'
    r'|migrat(?:e[sd]?|ing)\s+(?:from|away))\b',
    re.IGNORECASE,
)
_RE_USES = re.compile(
    r'\b(?:uses?|using|import[sd]?|require[sd]?|depends?\s+on|calls?|invokes?)\b',
    re.IGNORECASE,
)
_RE_SOLVES = re.compile(
    r'\b(?:fix(?:es|ed)?|solves?|resolves?|work(?:s|ed)?\s+around|patch(?:es|ed)?)\b',
    re.IGNORECASE,
)
_RE_CONFLICTS = re.compile(
    r'\b(?:conflicts?\s+with|breaks?|incompatible|clash(?:es)?)\b',
    re.IGNORECASE,
)


def extract_relations(
    content: str,
    entities: list[ExtractedEntity],
) -> list[ExtractedRelation]:
    """Infer relationships between extracted entities based on co-occurrence and signals."""
    if len(entities) < 2:
        return []

    relations: list[ExtractedRelation] = []
    seen_rels: set[tuple[str, str, str]] = set()

    def _add_rel(src: str, tgt: str, rel: str, conf: float = 1.0):
        key = (src, tgt, rel)
        if key not in seen_rels and src != tgt:
            seen_rels.add(key)
            relations.append(ExtractedRelation(src, tgt, rel, conf))

    # Split into sentences for co-occurrence
    sentences = _RE_SENTENCE.split(content)
    sent_offset = 0

    for sent in sentences:
        sent_end = sent_offset + len(sent)

        # Find entities in this sentence, ordered by position in text
        sent_entities = []
        for e in entities:
            if e.span and e.span[0] >= sent_offset and e.span[1] <= sent_end:
                sent_entities.append((e.span[0], e))
            elif e.name in sent.lower():
                # Use position of name in sentence for ordering
                pos = sent.lower().find(e.name)
                sent_entities.append((sent_offset + pos if pos >= 0 else sent_end, e))
        # Sort by position to get stable source/target directionality
        sent_entities.sort(key=lambda x: x[0])
        sent_entities = [e for _, e in sent_entities]

        # Pairwise relation inference within sentence
        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i + 1:]:
                # Check for specific relation signals
                if _RE_SUPERSEDES.search(sent):
                    _add_rel(e1.name, e2.name, "supersedes", 0.8)
                elif _RE_SOLVES.search(sent):
                    if e2.entity_type == "error":
                        _add_rel(e1.name, e2.name, "solves", 0.9)
                    elif e1.entity_type == "error":
                        _add_rel(e2.name, e1.name, "solves", 0.9)
                    else:
                        _add_rel(e1.name, e2.name, "solves", 0.7)
                elif _RE_CONFLICTS.search(sent):
                    _add_rel(e1.name, e2.name, "conflicts_with", 0.8)
                elif _RE_USES.search(sent):
                    _add_rel(e1.name, e2.name, "uses", 0.7)
                else:
                    _add_rel(e1.name, e2.name, "related_to", 0.5)

        sent_offset = sent_end + 1  # +1 for the split delimiter

    # Directory containment: file entities sharing a prefix
    # Collect directories that need materialization
    dirs_seen: set[str] = set()
    file_entities = [e for e in entities if e.entity_type == "file" and "/" in e.name]
    for i, f1 in enumerate(file_entities):
        for f2 in file_entities[i + 1:]:
            dir1 = f1.name.rsplit("/", 1)[0]
            dir2 = f2.name.rsplit("/", 1)[0]
            if dir1 == dir2:
                dirs_seen.add(dir1)
                _add_rel(dir1, f1.name, "contains", 0.9)
                _add_rel(dir1, f2.name, "contains", 0.9)

    # Materialize directory entities so store_entities_and_edges can resolve IDs
    for d in dirs_seen:
        canon = _normalize_name(d, "file")
        key = (canon, "file")
        if key not in {(e.name, e.entity_type) for e in entities}:
            entities.append(ExtractedEntity(
                name=canon,
                display_name=d,
                entity_type="file",
                metadata={"is_directory": True},
            ))

    return relations


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------


async def store_entities_and_edges(
    memory_id: str,
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
) -> dict:
    """Persist entities and edges to PostgreSQL.

    Idempotent per memory_id: re-processing the same learning will not inflate
    mention_count or edge weight. Uses kg_entity_mentions as the dedup gate —
    entities and edges are only counted when a genuinely new mention is inserted.
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    entity_count = 0
    edge_count = 0
    mention_count = 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Map (canonical_name, entity_type) -> entity UUID for edge insertion
            name_type_to_id: dict[tuple[str, str], str] = {}
            # Also keep name-only lookup for edge resolution (first match)
            name_to_id: dict[str, str] = {}

            # Upsert entities — get or create, but do NOT increment mention_count yet
            for e in entities:
                row = await conn.fetchrow(
                    """
                    INSERT INTO kg_entities (name, display_name, entity_type, metadata)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (name, entity_type) DO UPDATE SET
                        last_seen_at = NOW()
                    RETURNING id
                    """,
                    e.name,
                    e.display_name,
                    e.entity_type,
                    "{}",
                )
                if row:
                    eid = str(row["id"])
                    name_type_to_id[(e.name, e.entity_type)] = eid
                    if e.name not in name_to_id:
                        name_to_id[e.name] = eid
                    entity_count += 1

            # Insert entity mentions — the dedup gate.
            # Only increment mention_count when a new row is actually inserted.
            for e in entities:
                eid = name_type_to_id.get((e.name, e.entity_type))
                if not eid:
                    continue
                inserted = await conn.fetchval(
                    """
                    INSERT INTO kg_entity_mentions (entity_id, memory_id, mention_type,
                                                     span_start, span_end)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5)
                    ON CONFLICT (entity_id, memory_id) DO NOTHING
                    RETURNING TRUE
                    """,
                    eid,
                    memory_id,
                    "explicit",
                    e.span[0] if e.span else None,
                    e.span[1] if e.span else None,
                )
                if inserted:
                    mention_count += 1
                    # Only bump mention_count on genuinely new mention
                    await conn.execute(
                        """
                        UPDATE kg_entities SET mention_count = mention_count + 1
                        WHERE id = $1::uuid
                        """,
                        eid,
                    )

            # Upsert edges — only add weight for new (memory_id, source, target, relation)
            for rel in relations:
                src_id = name_to_id.get(rel.source)
                tgt_id = name_to_id.get(rel.target)
                if not src_id or not tgt_id:
                    continue
                # Check if this exact edge already exists for this memory_id
                existing = await conn.fetchval(
                    """
                    SELECT 1 FROM kg_edges
                    WHERE source_id = $1::uuid AND target_id = $2::uuid
                      AND relation = $3 AND memory_id = $4::uuid
                    """,
                    src_id,
                    tgt_id,
                    rel.relation,
                    memory_id,
                )
                if existing:
                    continue
                await conn.execute(
                    """
                    INSERT INTO kg_edges (source_id, target_id, relation, weight, memory_id)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5::uuid)
                    ON CONFLICT (source_id, target_id, relation) DO UPDATE SET
                        weight = kg_edges.weight + $4,
                        updated_at = NOW()
                    """,
                    src_id,
                    tgt_id,
                    rel.relation,
                    rel.confidence,
                    memory_id,
                )
                edge_count += 1

    return {
        "entities": entity_count,
        "edges": edge_count,
        "mentions": mention_count,
    }
