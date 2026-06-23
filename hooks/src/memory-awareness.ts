/*!
 * Memory Awareness Hook (UserPromptSubmit)
 *
 * Checks if user prompt is similar to stored learnings.
 * Shows hint to BOTH user (visible) AND Claude (system context).
 *
 * Flow:
 * 1. Extract INTENT from user prompt (not just keywords)
 * 2. Semantic search using hybrid RRF (text + vector)
 * 3. If score > threshold, show visible hint with top learning preview
 * 4. Claude proactively discloses and acts on relevant memories
 */

import { readFileSync, existsSync } from 'fs';
import { spawnSync } from 'child_process';
import { getOpcDir } from './shared/opc-path.js';
import { runPgQuery } from './shared/db-utils-pg.js';

/**
 * Upper bound on the per-session surfaced-id set (issue #228 item 2).
 * Bounds the recall argv and the persisted array so a long session can't grow
 * the exclusion list without limit.
 */
export const SURFACED_ID_CAP = 500;

interface UserPromptSubmitInput {
  session_id: string;
  hook_event_name: string;
  prompt: string;
  cwd: string;
}

interface LearningResult {
  id: string;
  type: string;
  content: string;
  score: number;
}

interface MemoryMatch {
  count: number;
  results: LearningResult[];
  // Full (untruncated) UUIDs of the surfaced results, for the per-session
  // exclusion union (issue #228 item 2). LearningResult.id is sliced to 8
  // chars for display, so the union must use this separate full-id list.
  fullIds: string[];
}

function readStdin(): string {
  return readFileSync(0, 'utf-8');
}

/**
 * Extract the INTENT from user prompt - what they're actually asking about.
 * Removes meta-language ("can you", "help me", "recall") to get core topic.
 */
function extractIntent(prompt: string): string {
  // Meta-phrases to remove (these describe HOW, not WHAT)
  const metaPhrases = [
    /^(can you|could you|would you|please|help me|i want to|i need to|let's|lets)\s+/gi,
    /^(show me|tell me|find|search for|look for|recall|remember)\s+/gi,
    /^(how do i|how can i|how to|what is|what are|where is|where are)\s+/gi,
    /\s+(for me|please|thanks|thank you)$/gi,
    /\?$/g,
  ];

  let intent = prompt.trim();

  // Strip meta-phrases iteratively
  for (const pattern of metaPhrases) {
    intent = intent.replace(pattern, '');
  }

  intent = intent.trim();

  // If we stripped too much, fall back to keyword extraction
  if (intent.length < 5) {
    return extractKeywords(prompt);
  }

  return intent;
}

/**
 * Extract meaningful keywords from prompt (fallback for very short intents).
 */
function extractKeywords(prompt: string): string {
  const stopWords = new Set([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'just', 'don', 'now', 'i', 'me', 'my', 'you', 'your', 'we', 'help', 'with',
    'our', 'they', 'them', 'their', 'it', 'its', 'this', 'that', 'these',
    'what', 'which', 'who', 'whom', 'and', 'but', 'if', 'or', 'because',
    'until', 'while', 'about', 'against', 'also', 'get', 'got', 'make',
    'want', 'need', 'look', 'see', 'use', 'like', 'know', 'think', 'take',
    'come', 'go', 'say', 'said', 'tell', 'please', 'help', 'let', 'sure',
    'recall', 'remember', 'similar', 'problems', 'issues'
  ]);

  const words = prompt
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !stopWords.has(w));

  return [...new Set(words)].slice(0, 5).join(' ');
}

/**
 * Normalize an intent string into a Postgres FTS query (issue #213).
 *
 * Underscores/slashes become spaces and runs of whitespace collapse, but we
 * deliberately DO NOT strip short tokens: the previous `\b\w{1,2}\b` removal
 * discarded bare numbers ("7") and 2-char tech tokens ("os", "pg", "v2", "ci"),
 * collapsing a specific prompt into a generic phrase that then matched
 * unrelated cross-project learnings. `plainto_tsquery` already drops English
 * stopwords downstream, so the blanket strip only destroyed signal.
 */
export function sanitizeSearchTerm(intent: string): string {
  return intent
    .replace(/[_\/]/g, ' ')           // Convert underscores/slashes to spaces
    .replace(/\s+/g, ' ')             // Collapse whitespace
    .trim();
}

// Lead tokens that mark a conversational reply rather than a knowledge query.
const CONVERSATIONAL_LEAD = new Set([
  'no', 'nope', 'nah', 'yes', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure',
  'nvm', 'nevermind', 'oops', 'thanks', 'exactly', 'agreed',
  'cool', 'great', 'awesome', 'perfect'
]);

// Imperative on a *bare* pronoun that consumes the WHOLE remainder, e.g.
// "do it", "undo that", "run it again", "extend it another 7 days". The match
// is anchored to end-of-string: the pronoun may be followed only by meta
// continuations (again/now/instead/...) or a bounded numeric "another <n>
// <unit>" quantity ("another 7 days", "another 7 business days"). It is tested
// against a punctuation-stripped, whitespace-normalized token string, so the
// `another` branch requires a DIGIT — a generic noun tail like "another auth
// pattern" does NOT match. This deliberately falls through to recall for real
// noun phrases: "fix that bug", "change this function", "fix this instead with
// stored auth pattern", "update this now using the pg v2 note".
const PRONOUN_IMPERATIVE =
  /^(?:do|redo|undo|run|rerun|try|retry|repeat|revert|keep|continue|extend|fix|change|update|move|apply|test|save|delete|remove|show|add|create|make|use)\s+(?:it|that|this|them|those|these)(?:\s+(?:again|now|once|more|instead|please|too|another\s+\d+(?:\s+\w+){1,2}))*$/;

// Selection-style meta imperative, e.g. "do the second one", "do the next
// option", "use the other approach" — issue #213 lists these explicitly. The
// object is "the <ordinal|next|other|...>" optionally followed by a generic
// choice noun, anchored to end-of-string. A real noun phrase ("run the second
// test suite", "do the migration") does NOT match because the trailing words
// fall outside the bounded choice-noun set.
const SELECTION_IMPERATIVE =
  /^(?:do|redo|run|rerun|try|retry|repeat|use|pick|choose|select|take|apply|keep)\s+the\s+(?:first|second|third|fourth|fifth|sixth|last|next|previous|prior|other|another|latter|former|same|top|bottom|\d+(?:st|nd|rd|th)?)(?:\s+(?:one|ones|option|item|choice|approach|suggestion|result|match|idea|fix))?(?:\s+(?:again|now|please|instead|too))*$/;

/**
 * Decide whether a prompt is a short conversational/meta turn that should NOT
 * trigger a memory recall (issue #213).
 *
 * The always-on hook fired on `"no, extend it another 7 days"` — a meta-command
 * about the live conversation — and surfaced unrelated cross-project learnings.
 * The gate is intentionally biased toward LETTING prompts through: a missed
 * banner is cheap (the user can /recall), but a wrong cross-project banner is
 * noisy and erodes trust. We therefore strip any leading discourse marker and
 * classify the REMAINDER, gating only when:
 *  - the prompt is empty / pure affirmation ("yes", "no thanks", "ok sure");
 *  - the remainder is a pronoun-imperative consuming the whole tail
 *    ("do it", "yeah undo that", "no, extend it another 7 days");
 *  - the remainder is a selection-style imperative ("do the second one").
 * A real query after a marker ("no, explain pg pool leak") keeps its body and
 * is NOT gated. Prompts over 8 tokens are treated as substantive outright.
 */
export function isConversationalTurn(prompt: string): boolean {
  const trimmed = prompt.trim();
  if (!trimmed) return true;

  const lower = trimmed.toLowerCase();
  // De-glue a leading conversational marker from a punctuation delimiter that
  // has no following space ("no:do that", "no-do that", "no.do that") so the
  // marker is recognized regardless of delimiter or spacing. Only the lead
  // position is rewritten, so internal identifiers ("session-start",
  // "memory_daemon.py") and decimals elsewhere are left intact.
  const deglued = lower.replace(
    /^([a-z]+)\s*[,:;.\-]+\s*/,
    (match, word) => (CONVERSATIONAL_LEAD.has(word) ? `${word} ` : match)
  );
  const tokens = deglued
    .split(/\s+/)
    .map(t => t.replace(/^[^\w]+|[^\w]+$/g, ''))
    .filter(Boolean);
  if (tokens.length === 0) return true;
  if (tokens.length > 8) return false;  // substantive turn — let recall run

  // Pure acknowledgement: every token is a conversational lead word
  // ("yes", "no thanks", "ok sure", "nope nvm").
  if (tokens.every(t => CONVERSATIONAL_LEAD.has(t))) return true;

  // Drop a single leading discourse marker ("yeah do that" -> "do that";
  // "no, extend it..." -> "extend it..."), then require the REMAINDER to be a
  // whole-tail pronoun-imperative. The remainder is rebuilt from the already
  // punctuation-stripped tokens, so any lead delimiter (comma, colon, dash,
  // period) is handled uniformly. A substantive body ("no, explain pg pool
  // leak") survives the drop and is not gated.
  const body = (CONVERSATIONAL_LEAD.has(tokens[0]) ? tokens.slice(1) : tokens).join(' ');
  return PRONOUN_IMPERATIVE.test(body) || SELECTION_IMPERATIVE.test(body);
}

/**
 * Capture the FULL untruncated learning UUIDs from recall results for the
 * surfaced-id union (issue #228 item 2). The display path slices ids to 8
 * chars for the hint; this MUST use the full id so exclusion works next turn.
 */
export function extractFullIds(results: any[]): string[] {
  if (!Array.isArray(results)) return [];
  return results
    .map((r) => (r && typeof r.id === 'string' ? r.id : ''))
    .filter((id) => id.length > 0);
}

/**
 * Build the `--exclude-ids <uuid> ...` argv fragment, or [] (flag omitted)
 * when there is nothing to exclude.
 */
export function buildExcludeArgs(ids: string[]): string[] {
  if (!ids || ids.length === 0) return [];
  return ['--exclude-ids', ...ids];
}

/**
 * Dedupe-union prior surfaced ids with freshly returned ids and bound the set
 * to `cap`. Freshly surfaced ids are kept preferentially (they sit at the tail
 * and the cap trims from the head) so the most recent picks always stay
 * excluded.
 */
export function unionCap(prior: string[], fresh: string[], cap: number): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const id of [...(prior || []), ...(fresh || [])]) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    merged.push(id);
  }
  if (merged.length <= cap) return merged;
  // Trim from the head; the tail (most-recently-surfaced) survives.
  return merged.slice(merged.length - cap);
}

/**
 * Read prior surfaced learning ids for this session (issue #228 item 2).
 * Best-effort: any DB error / missing row / NULL column yields []. Keyed by
 * claude_session_id (= the hook's stdin input.session_id).
 */
export function readSurfacedIds(sessionId: string): string[] {
  if (!sessionId) return [];
  const pythonCode = `
import asyncpg, json

db_url = os.environ['CONTINUOUS_CLAUDE_DB_URL']
session_id = sys.argv[1]

async def main():
    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow(
            "SELECT surfaced_learning_ids FROM sessions WHERE claude_session_id = $1",
            session_id,
        )
    finally:
        await conn.close()
    ids = row['surfaced_learning_ids'] if row and row['surfaced_learning_ids'] else []
    print(json.dumps([str(x) for x in ids]))

asyncio.run(main())
`;
  try {
    const res = runPgQuery(pythonCode, [sessionId]);
    if (!res.success || !res.stdout) return [];
    const parsed = JSON.parse(res.stdout);
    if (!Array.isArray(parsed)) return [];
    const ids = parsed.filter((x): x is string => typeof x === 'string' && x.length > 0);
    // Defensive bound: a pre-existing oversized row (schema drift, or rows
    // written before the cap was enforced) must not produce an unbounded
    // --exclude-ids argv. Keep the most-recent tail, consistent with unionCap.
    return ids.length > SURFACED_ID_CAP ? ids.slice(-SURFACED_ID_CAP) : ids;
  } catch {
    return [];
  }
}

/**
 * Persist the session's surfaced set (issue #228 item 2). The caller passes the
 * complete, already-deduped-and-capped array (via unionCap), so the write
 * REPLACES the column rather than re-unioning with the existing value — a
 * re-union would re-add ids that unionCap trimmed and let the array grow without
 * bound. Per-session hook invocations are serial, so a plain replace is safe.
 * Best-effort: caps defensively and swallows all DB errors so the hook never
 * breaks.
 */
export function persistSurfacedIds(sessionId: string, ids: string[]): void {
  if (!sessionId || !ids || ids.length === 0) return;
  // Keep the most-recent tail if somehow over cap (matches unionCap semantics).
  const capped = ids.length > SURFACED_ID_CAP ? ids.slice(-SURFACED_ID_CAP) : ids;
  const pythonCode = `
import asyncpg, json

db_url = os.environ['CONTINUOUS_CLAUDE_DB_URL']
session_id = sys.argv[1]
ids = json.loads(sys.argv[2])

async def main():
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "UPDATE sessions SET surfaced_learning_ids = $2::uuid[] "
            "WHERE claude_session_id = $1",
            session_id,
            ids,
        )
    finally:
        await conn.close()

asyncio.run(main())
`;
  try {
    runPgQuery(pythonCode, [sessionId, JSON.stringify(capped)]);
  } catch {
    // Best-effort: never break the hook on a persistence failure.
  }
}

/**
 * Fast memory relevance check using text search.
 * For text-only mode, we search by the most significant keyword
 * (text ILIKE looks for substring match, not multi-word).
 */
function checkMemoryRelevance(
  intent: string,
  projectDir: string,
  excludeIds: string[] = []
): MemoryMatch | null {
  if (!intent || intent.length < 3) return null;

  const opcDir = getOpcDir();
  if (!opcDir) return null;  // Graceful degradation if OPC not available

  const searchTerm = sanitizeSearchTerm(intent);
  // An intent of only stripped chars (e.g. "___") sanitizes to empty — don't
  // spawn a recall process for a no-op query.
  if (!searchTerm) return null;

  // Derive project tag from CLAUDE_PROJECT_DIR to boost relevance via --tags (not a hard filter)
  const projectTag = projectDir ? projectDir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? '' : '';
  // Guard against tags starting with "-" to avoid them being parsed as CLI options
  const safeProjectTag = projectTag && !projectTag.startsWith('-') ? projectTag : '';

  // Use text-only for fast checking (< 1s), user can run /recall for semantic
  const tagArgs = safeProjectTag ? ['--tags', safeProjectTag] : [];
  // --project feeds the reranker's project_match signal (issue #130: the
  // hook previously never passed it, so the 0.15 project weight was always
  // zero on this path). Soft boost, not a filter.
  // --project-first (issue #139) scopes fetch-time retrieval: own-project
  // rows are fetched before the pool fills with big-project content — the
  // only remediation that works for small projects. Still fills globally,
  // and degrades to a plain global fetch on pre-migration DBs.
  const projectArgs = safeProjectTag
    ? ['--project', safeProjectTag, '--project-first']
    : [];
  // Hybrid search (issue #53, unblocked by #151): the corpus is single-space
  // voyage-code-3 and recall filters the vector leg by embedding_model, so
  // hybrid finally surfaces USER_PREFERENCE learnings that text-only FTS
  // misses on short prompts. --provider voyage is an API embed (~150-300ms,
  // no model cold-start); recall degrades to text-only within
  // QUERY_EMBED_TIMEOUT=2s if the key is missing or the provider stalls, so
  // worst case equals the old --text-only behavior. Hybrid also activates
  // the reranker's type-affinity signal (dead on the text-only path).
  // --source labels recall_log rows (issue #140 telemetry).
  const result = spawnSync('uv', [
    'run', 'python', 'scripts/core/recall_learnings.py',
    '--query', searchTerm,
    '--k', '3',
    '--json',
    '--provider', 'voyage',
    '--source', 'hook',
    ...tagArgs,
    ...projectArgs,
    // Issue #228 item 2: drop already-surfaced learnings BEFORE rank so the
    // hook stops re-surfacing the same top memories every turn. Appended to
    // the argv array (not shell) so full UUIDs are passed safely.
    ...buildExcludeArgs(excludeIds)
  ], {
    encoding: 'utf-8',
    cwd: opcDir,
    env: {
      ...process.env,
      // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71).
      UV_FROZEN: '1',
      PYTHONPATH: opcDir
    },
    timeout: 5000  // 5s timeout for fast check
  });

  if (result.status !== 0 || !result.stdout) {
    return null;
  }

  try {
    const data = JSON.parse(result.stdout);

    if (!data.results || data.results.length === 0) {
      return null;
    }

    // ts_rank returns small values (0.0001-0.1), ILIKE fallback returns 0.1
    // Any match from FTS is relevant enough to show

    // Extract structured results with better previews
    const results: LearningResult[] = data.results.slice(0, 3).map((r: any) => {
      const content = r.content || '';
      // Get first meaningful line up to 120 chars
      const preview = content
        .split('\n')
        .filter((l: string) => l.trim().length > 0)
        .map((l: string) => l.trim())
        .join(' ')
        .slice(0, 120);

      return {
        id: (r.id || 'unknown').slice(0, 8),
        type: r.learning_type || r.type || 'UNKNOWN',
        content: preview + (content.length > 120 ? '...' : ''),
        score: r.score || 0
      };
    });

    return {
      count: data.results.length,
      results,
      // Capture the FULL untruncated ids (not the 8-char display slice above)
      // for the per-session exclusion union (issue #228 item 2).
      fullIds: extractFullIds(data.results)
    };
  } catch {
    return null;
  }
}

async function main() {
  const input: UserPromptSubmitInput = JSON.parse(readStdin());
  const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd;

  // Skip for subagents - they don't need memory recall (saves tokens)
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }

  // Skip very short prompts (greetings, commands)
  if (input.prompt.length < 15) {
    return;
  }

  // Skip if prompt is just a slash command
  if (input.prompt.trim().startsWith('/')) {
    return;
  }

  // Skip short conversational/meta turns (issue #213): replies like
  // "no, extend it another 7 days" carry no archival intent and otherwise
  // recall unrelated, often cross-project, learnings.
  if (isConversationalTurn(input.prompt)) {
    return;
  }

  // Extract intent (semantic query, not just keywords)
  const intent = extractIntent(input.prompt);

  // Skip if no meaningful intent
  if (intent.length < 3) {
    return;
  }

  // Issue #228 item 2: read learnings already surfaced this session so recall
  // can exclude them BEFORE ranking (stops re-surfacing the same top memories
  // every turn). Best-effort — readSurfacedIds swallows all DB errors to [].
  const priorSurfaced = readSurfacedIds(input.session_id);

  // Check memory relevance using semantic search
  const match = checkMemoryRelevance(intent, projectDir, priorSurfaced);

  if (match) {
    // Union the freshly surfaced full UUIDs into the session's surfaced set so
    // next turn excludes them too. Best-effort; capped and error-swallowing.
    persistSurfacedIds(
      input.session_id,
      unionCap(priorSurfaced, match.fullIds, SURFACED_ID_CAP)
    );

    // Build structured context for Claude
    const resultLines = match.results.map((r, i) =>
      `${i + 1}. [${r.type}] ${r.content} (id: ${r.id})`
    ).join('\n');

    const claudeContext = `MEMORY MATCH (${match.count} results) for "${intent}":\n${resultLines}\nUse /recall "${intent}" for full content. Disclose if helpful.`;

    console.log(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: 'UserPromptSubmit',
        additionalContext: claudeContext
      }
    }));
  }
}

// Only run as the hook entry point — guard so the pure helpers above can be
// imported in unit tests without main() consuming stdin (matches the
// convention in heartbeat.ts / working-on-sync.ts).
if (
  typeof process !== 'undefined' &&
  process.argv[1] &&
  (process.argv[1].endsWith('memory-awareness.ts') ||
    process.argv[1].endsWith('memory-awareness.js') ||
    process.argv[1].endsWith('memory-awareness.mjs'))
) {
  main().catch(() => {
    // Silent fail - don't block user prompts
  });
}
