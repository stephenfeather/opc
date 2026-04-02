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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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
_IDF_MAX_AGE_HOURS = 24
_IDF_DRIFT_THRESHOLD = 0.10  # rebuild if doc count drifts >10%


def _tokenize(text: str) -> list[str]:
    """Tokenize text for IDF computation.

    Lowercase, replace hyphens with spaces, split, strip non-alnum,
    filter short words and stopwords.
    """
    text = text.lower().replace("-", " ")
    words = []
    for w in text.split():
        w = _ALNUM_RE.sub("", w)
        if len(w) > 2 and w not in STOPWORDS:
            words.append(w)
    return words


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

    Uses a cursor to stream rows instead of loading the entire corpus into RAM.

    Args:
        path: Where to save the index. Defaults to _IDF_CACHE_PATH.
    """
    from scripts.core.db.postgres_pool import get_pool

    pool = await get_pool()
    word_df: dict[str, int] = {}
    doc_count = 0

    async with pool.acquire() as conn:
        async with conn.transaction():
            async for row in conn.cursor(
                """
                SELECT content FROM archival_memory
                WHERE metadata->>'type' = 'session_learning'
                AND superseded_by IS NULL
                """
            ):
                doc_count += 1
                unique_words = set(_tokenize(row["content"]))
                for w in unique_words:
                    word_df[w] = word_df.get(w, 0) + 1

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
    - Cache is older than 24 hours
    - Doc count has drifted >10% from cached count
    """
    if not force_rebuild:
        cached = load_idf_index(path)
        if cached is not None:
            # Check age
            try:
                built = datetime.fromisoformat(cached.built_at)
                age_hours = (datetime.now(UTC) - built).total_seconds() / 3600
                if age_hours < _IDF_MAX_AGE_HOURS:
                    # Check drift
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
                    if cached.doc_count > 0:
                        drift = abs(current_count - cached.doc_count) / cached.doc_count
                    else:
                        drift = 1.0
                    if drift <= _IDF_DRIFT_THRESHOLD:
                        return cached
                    logger.info(
                        "IDF index drift %.1f%% exceeds threshold, rebuilding",
                        drift * 100,
                    )
            except (ValueError, TypeError):
                pass  # bad timestamp, rebuild

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

    # Build original query terms for the OR string
    original_tokens = set(_tokenize(query))
    # Sanitize original words for tsquery safety (strip FTS operators and non-alnum)
    original_words = []
    for w in query.lower().replace("-", " ").split():
        clean = re.sub(r"[^a-zA-Z0-9]", "", w)
        if len(clean) > 2:
            original_words.append(clean)
    if not original_words:
        # Last resort: take first word, cleaned
        fallback = re.sub(r"[^a-zA-Z0-9]", "", query.lower().split()[0]) if query.strip() else ""
        original_words = [fallback] if fallback else [""]

    # Fetch vector neighbors (pool init registers pgvector codec per connection)
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
        return " | ".join(original_words)

    # Tokenize neighbors and count doc frequency within neighbor set
    neighbor_df: dict[str, int] = {}
    for row in rows:
        unique_words = set(_tokenize(row["content"]))
        for w in unique_words:
            neighbor_df[w] = neighbor_df.get(w, 0) + 1

    # Get corpus IDF
    idf_index = await get_idf_index()

    # Score: neighbor_df * corpus_idf
    # High score = common among neighbors AND rare in corpus
    candidates: list[tuple[str, float]] = []
    for term, ndf in neighbor_df.items():
        if term in original_tokens:
            continue
        if term in STOPWORDS or len(term) <= 2:
            continue
        corpus_idf = idf_index.idf(term)
        if corpus_idf < min_idf:
            continue
        # Sanitize for tsquery safety
        clean = re.sub(r"[^a-zA-Z0-9]", "", term)
        if not clean or len(clean) <= 2:
            continue
        score = ndf * corpus_idf
        candidates.append((clean, score))

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    expansion_terms = [t for t, _ in candidates[:max_expansion_terms]]

    # Build OR-joined tsquery string
    all_terms = original_words + expansion_terms
    return " | ".join(all_terms)
