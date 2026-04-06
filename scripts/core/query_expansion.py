"""TF-IDF query expansion for hybrid RRF recall.

Expands text queries with semantically related terms before hybrid search.
Uses vector neighbors + corpus IDF to find high-signal expansion terms.

Example: "auth patterns" -> "auth | patterns | authentication | workflow | tokens"
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from scripts.core.config import get_config as _get_config

logger = logging.getLogger(__name__)

# Shared stopwords — merges recall_backends meta_words with standard English
STOPWORDS: frozenset[str] = frozenset({
    # Meta-words from recall_backends.py
    "help", "want", "need", "show", "tell", "find", "look", "please", "with", "for",
    # Standard English stopwords
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "is", "it",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "just", "about", "above", "after", "again",
    "also", "any", "because", "before", "below", "between", "by", "down", "during",
    "from", "further", "get", "got", "here", "into", "its", "let", "like", "make",
    "me", "my", "myself", "now", "off", "once", "our", "out", "over", "still",
    "there", "their", "them", "they", "through", "under", "until", "up", "upon",
    "use", "used", "using", "we", "well", "your", "you",
})

_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")
_IDF_CACHE_PATH = Path.home() / ".claude" / "cache" / "idf_index.json"
_qe_cfg = _get_config().query_expansion
_IDF_MAX_AGE_HOURS = _qe_cfg.idf_max_age_hours
_IDF_DRIFT_THRESHOLD = _qe_cfg.idf_drift_threshold


# --- Pure functions ---


def _tokenize(text: str) -> list[str]:
    """Tokenize text for IDF computation.

    Lowercase, replace hyphens with spaces, split, strip non-alnum,
    filter short words and stopwords.
    """
    text = text.lower().replace("-", " ")
    return [
        clean
        for w in text.split()
        if len(clean := _ALNUM_RE.sub("", w)) > 2 and clean not in STOPWORDS
    ]


def _sanitize_query_words(query: str) -> list[str]:
    """Extract and sanitize query words for tsquery output.

    Lowercases, replaces hyphens, strips non-alnum, filters short words.
    Falls back to first cleaned word (or empty string) when no words survive.
    """
    words = [
        clean
        for w in query.lower().replace("-", " ").split()
        if len(clean := re.sub(r"[^a-zA-Z0-9]", "", w)) > 2
    ]
    if words:
        return words
    # Fallback: take first word cleaned, or empty string
    fallback = re.sub(r"[^a-zA-Z0-9]", "", query.lower().split()[0]) if query.strip() else ""
    return [fallback] if fallback else [""]


def _compute_neighbor_df(contents: Iterable[str]) -> dict[str, int]:
    """Compute document frequency of tokens across neighbor documents.

    Each document contributes at most once per unique token (set-based).
    """
    df: dict[str, int] = {}
    for text in contents:
        for w in set(_tokenize(text)):
            df[w] = df.get(w, 0) + 1
    return df


def _compute_word_df(documents: Iterable[str]) -> tuple[dict[str, int], int]:
    """Compute corpus-wide word document frequency.

    Returns (word_df dict, document count).
    """
    word_df: dict[str, int] = {}
    doc_count = 0
    for text in documents:
        doc_count += 1
        for w in set(_tokenize(text)):
            word_df[w] = word_df.get(w, 0) + 1
    return word_df, doc_count


def _score_expansion_candidates(
    neighbor_df: dict[str, int],
    idf_index: IDFIndex,
    original_tokens: set[str],
    min_idf: float,
) -> list[tuple[str, float]]:
    """Score and filter expansion candidates by neighbor_df * corpus_idf.

    Excludes original tokens, stopwords, short terms, and low-IDF terms.
    Returns sorted list of (term, score) descending by score.
    """
    candidates: list[tuple[str, float]] = []
    for term, ndf in neighbor_df.items():
        if term in original_tokens or term in STOPWORDS or len(term) <= 2:
            continue
        corpus_idf = idf_index.idf(term)
        if corpus_idf < min_idf:
            continue
        clean = re.sub(r"[^a-zA-Z0-9]", "", term)
        if not clean or len(clean) <= 2:
            continue
        candidates.append((clean, ndf * corpus_idf))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _format_tsquery(original_words: list[str], expansion_terms: list[str]) -> str:
    """Join original and expansion terms into an OR-joined tsquery string."""
    all_terms = original_words + expansion_terms
    return " | ".join(all_terms)


def _is_cache_stale(
    cached: IDFIndex,
    *,
    max_age_hours: float,
    current_count: int,
    drift_threshold: float,
) -> bool:
    """Determine whether a cached IDF index needs rebuilding.

    Stale when: age exceeds max_age_hours, doc count drift exceeds threshold,
    zero cached docs, or unparseable timestamp.
    """
    try:
        built = datetime.fromisoformat(cached.built_at)
        age_hours = (datetime.now(UTC) - built).total_seconds() / 3600
        if age_hours >= max_age_hours:
            return True
    except (ValueError, TypeError):
        return True

    if cached.doc_count == 0:
        return True

    drift = abs(current_count - cached.doc_count) / cached.doc_count
    return drift > drift_threshold


# --- Data structures ---


@dataclass
class IDFIndex:
    """Inverse Document Frequency index over the learnings corpus."""

    word_df: dict[str, int] = field(default_factory=dict)
    doc_count: int = 0
    built_at: str = ""

    def idf(self, word: str) -> float:
        """Compute smoothed IDF for a word. Unknown words get max IDF."""
        df = self.word_df.get(word, 0)
        if df == 0:
            return math.log(self.doc_count + 1)  # max IDF for unknown
        return math.log(self.doc_count / (1 + df))


# --- I/O boundary functions ---


def save_idf_index(index: IDFIndex, path: Path | None = None) -> None:
    """Persist IDF index to JSON."""
    path = path or _IDF_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "word_df": index.word_df,
        "doc_count": index.doc_count,
        "built_at": index.built_at,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def load_idf_index(path: Path | None = None) -> IDFIndex | None:
    """Load cached IDF index from JSON. Returns None if missing/corrupt."""
    path = path or _IDF_CACHE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return IDFIndex(
            word_df=data["word_df"],
            doc_count=data["doc_count"],
            built_at=data["built_at"],
        )
    except (json.JSONDecodeError, KeyError):
        logger.warning("Corrupt IDF index at %s, will rebuild", path)
        return None


async def build_idf_index(path: Path | None = None) -> IDFIndex:
    """Build IDF index from all active learnings in the database.

    Streams rows via cursor, delegates counting to _compute_word_df.
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    documents: list[str] = []

    async with pool.acquire() as conn:
        async with conn.transaction():
            async for row in conn.cursor(
                """
                SELECT content FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                AND superseded_by IS NULL
                """
            ):
                documents.append(row["content"])

    word_df, doc_count = _compute_word_df(documents)

    index = IDFIndex(
        word_df=word_df,
        doc_count=doc_count,
        built_at=datetime.now(UTC).isoformat(),
    )
    save_idf_index(index, path)
    logger.info("Built IDF index: %d docs, %d terms", doc_count, len(word_df))
    return index


async def get_idf_index(
    force_rebuild: bool = False, path: Path | None = None
) -> IDFIndex:
    """Load cached IDF index or rebuild if missing/stale.

    Rebuilds when:
    - No cached index exists
    - Cache is older than configured max age
    - Doc count has drifted beyond configured threshold
    """
    if not force_rebuild:
        cached = load_idf_index(path)
        if cached is not None:
            from scripts.core.db.postgres_pool import get_pool

            pool = await get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as cnt FROM archival_memory
                    WHERE metadata->>'type' = 'session_learning'
                    AND superseded_by IS NULL
                    """
                )
            current_count = row["cnt"] if row else 0

            if not _is_cache_stale(
                cached,
                max_age_hours=_IDF_MAX_AGE_HOURS,
                current_count=current_count,
                drift_threshold=_IDF_DRIFT_THRESHOLD,
            ):
                return cached
            logger.info("IDF cache stale, rebuilding")

    return await build_idf_index(path)


async def expand_query(
    query: str,
    query_embedding: list[float],
    *,
    max_neighbors: int = 20,
    max_expansion_terms: int = 5,
    min_idf: float = 1.0,
) -> str:
    """Expand a query with related terms from vector neighbors.

    Finds vector-similar documents, extracts terms that are common
    among neighbors but rare in the corpus (high TF-IDF signal).

    Args:
        query: Original search query
        query_embedding: Pre-computed embedding for the query
        max_neighbors: Number of vector neighbors to examine
        max_expansion_terms: Max terms to add
        min_idf: Minimum corpus IDF to include a term

    Returns:
        OR-joined tsquery string: "original | term1 | term2 | ..."
    """
    from scripts.core.db.postgres_pool import get_pool

    original_tokens = set(_tokenize(query))
    original_words = _sanitize_query_words(query)

    # Fetch vector neighbors (I/O boundary)
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT content FROM archival_memory
            WHERE metadata->>'type' = 'session_learning'
            AND superseded_by IS NULL
            AND embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            str(query_embedding),
            max_neighbors,
        )

    if not rows:
        return _format_tsquery(original_words, [])

    # Pure logic: compute neighbor df, score candidates, format output
    neighbor_df = _compute_neighbor_df(row["content"] for row in rows)
    idf_index = await get_idf_index()
    candidates = _score_expansion_candidates(
        neighbor_df, idf_index, original_tokens, min_idf
    )
    expansion_terms = [t for t, _ in candidates[:max_expansion_terms]]

    return _format_tsquery(original_words, expansion_terms)
