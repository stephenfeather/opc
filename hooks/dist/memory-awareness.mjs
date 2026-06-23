// src/memory-awareness.ts
import { readFileSync as readFileSync2 } from "fs";
import { spawnSync as spawnSync2 } from "child_process";

// src/shared/opc-path.ts
import { existsSync, readFileSync } from "fs";
import { join } from "path";
function getOpcDirFromConfig() {
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (!homeDir) return null;
  const configPath = join(homeDir, ".claude", "opc.json");
  if (!existsSync(configPath)) return null;
  try {
    const content = readFileSync(configPath, "utf-8");
    const config = JSON.parse(content);
    const opcDir = config.opc_dir;
    if (opcDir && typeof opcDir === "string" && existsSync(opcDir)) {
      return opcDir;
    }
  } catch {
  }
  return null;
}
function getOpcDir() {
  const envOpcDir = process.env.CLAUDE_OPC_DIR;
  if (envOpcDir && existsSync(envOpcDir)) {
    return envOpcDir;
  }
  const configOpcDir = getOpcDirFromConfig();
  if (configOpcDir) {
    return configOpcDir;
  }
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const localOpc = join(projectDir, "opc");
  if (existsSync(localOpc)) {
    return localOpc;
  }
  const homeDir = process.env.HOME || process.env.USERPROFILE || "";
  if (homeDir) {
    const globalClaude = join(homeDir, ".claude");
    const globalScripts = join(globalClaude, "scripts", "core");
    if (existsSync(globalScripts)) {
      return globalClaude;
    }
  }
  return null;
}
function requireOpcDir() {
  const opcDir = getOpcDir();
  if (!opcDir) {
    console.log(JSON.stringify({ result: "continue" }));
    process.exit(0);
  }
  return opcDir;
}

// src/shared/db-utils-pg.ts
import { spawn, spawnSync } from "child_process";
function getPgConnectionString() {
  const url = process.env.CONTINUOUS_CLAUDE_DB_URL || process.env.DATABASE_URL || process.env.OPC_POSTGRES_URL;
  if (!url) {
    throw new Error(
      "Database URL not set. Set CONTINUOUS_CLAUDE_DB_URL (preferred), DATABASE_URL, or OPC_POSTGRES_URL. For local Docker dev, run `docker compose -f docker/docker-compose.yml up -d` and export the credentials from docker/.env before invoking this hook."
    );
  }
  return url;
}
function runPgQuery(pythonCode, args = []) {
  const opcDir = requireOpcDir();
  const resolvedDbUrl = getPgConnectionString();
  const wrappedCode = `
import sys
import os
import asyncio
import json

# Add opc to path for imports (read from env to avoid code injection)
_opc_dir = os.environ.get('_OPC_DIR')
if not _opc_dir:
    raise RuntimeError('_OPC_DIR environment variable not set - must be called via runPgQuery()')
sys.path.insert(0, _opc_dir)
os.chdir(_opc_dir)

${pythonCode}
`;
  try {
    const result = spawnSync("uv", ["run", "python", "-c", wrappedCode, ...args], {
      encoding: "utf-8",
      maxBuffer: 1024 * 1024,
      timeout: 5e3,
      // 5 second timeout - fail gracefully if DB unreachable
      cwd: opcDir,
      env: {
        ...process.env,
        // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71
        // follow-up); use the lock as-is. Intentional updates use `uv lock`.
        UV_FROZEN: "1",
        CONTINUOUS_CLAUDE_DB_URL: resolvedDbUrl,
        _OPC_DIR: opcDir
      }
    });
    return {
      success: result.status === 0,
      stdout: result.stdout?.trim() || "",
      stderr: result.stderr || ""
    };
  } catch (err) {
    return {
      success: false,
      stdout: "",
      stderr: String(err)
    };
  }
}

// src/memory-awareness.ts
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
var SURFACED_ID_CAP = 500;
function readStdin() {
  return readFileSync2(0, "utf-8");
}
function extractIntent(prompt) {
  const metaPhrases = [
    /^(can you|could you|would you|please|help me|i want to|i need to|let's|lets)\s+/gi,
    /^(show me|tell me|find|search for|look for|recall|remember)\s+/gi,
    /^(how do i|how can i|how to|what is|what are|where is|where are)\s+/gi,
    /\s+(for me|please|thanks|thank you)$/gi,
    /\?$/g
  ];
  let intent = prompt.trim();
  for (const pattern of metaPhrases) {
    intent = intent.replace(pattern, "");
  }
  intent = intent.trim();
  if (intent.length < 5) {
    return extractKeywords(prompt);
  }
  return intent;
}
function extractKeywords(prompt) {
  const stopWords = /* @__PURE__ */ new Set([
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "don",
    "now",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "help",
    "with",
    "our",
    "they",
    "them",
    "their",
    "it",
    "its",
    "this",
    "that",
    "these",
    "what",
    "which",
    "who",
    "whom",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "about",
    "against",
    "also",
    "get",
    "got",
    "make",
    "want",
    "need",
    "look",
    "see",
    "use",
    "like",
    "know",
    "think",
    "take",
    "come",
    "go",
    "say",
    "said",
    "tell",
    "please",
    "help",
    "let",
    "sure",
    "recall",
    "remember",
    "similar",
    "problems",
    "issues"
  ]);
  const words = prompt.toLowerCase().replace(/[^\w\s-]/g, " ").split(/\s+/).filter((w) => w.length > 2 && !stopWords.has(w));
  return [...new Set(words)].slice(0, 5).join(" ");
}
function sanitizeSearchTerm(intent) {
  return intent.replace(/[_\/]/g, " ").replace(/\s+/g, " ").trim();
}
var CONVERSATIONAL_LEAD = /* @__PURE__ */ new Set([
  "no",
  "nope",
  "nah",
  "yes",
  "yeah",
  "yep",
  "yup",
  "ok",
  "okay",
  "sure",
  "nvm",
  "nevermind",
  "oops",
  "thanks",
  "exactly",
  "agreed",
  "cool",
  "great",
  "awesome",
  "perfect"
]);
var PRONOUN_IMPERATIVE = /^(?:do|redo|undo|run|rerun|try|retry|repeat|revert|keep|continue|extend|fix|change|update|move|apply|test|save|delete|remove|show|add|create|make|use)\s+(?:it|that|this|them|those|these)(?:\s+(?:again|now|once|more|instead|please|too|another\s+\d+(?:\s+\w+){1,2}))*$/;
var SELECTION_IMPERATIVE = /^(?:do|redo|run|rerun|try|retry|repeat|use|pick|choose|select|take|apply|keep)\s+the\s+(?:first|second|third|fourth|fifth|sixth|last|next|previous|prior|other|another|latter|former|same|top|bottom|\d+(?:st|nd|rd|th)?)(?:\s+(?:one|ones|option|item|choice|approach|suggestion|result|match|idea|fix))?(?:\s+(?:again|now|please|instead|too))*$/;
function isConversationalTurn(prompt) {
  const trimmed = prompt.trim();
  if (!trimmed) return true;
  const lower = trimmed.toLowerCase();
  const deglued = lower.replace(
    /^([a-z]+)\s*[,:;.\-]+\s*/,
    (match, word) => CONVERSATIONAL_LEAD.has(word) ? `${word} ` : match
  );
  const tokens = deglued.split(/\s+/).map((t) => t.replace(/^[^\w]+|[^\w]+$/g, "")).filter(Boolean);
  if (tokens.length === 0) return true;
  if (tokens.length > 8) return false;
  if (tokens.every((t) => CONVERSATIONAL_LEAD.has(t))) return true;
  const body = (CONVERSATIONAL_LEAD.has(tokens[0]) ? tokens.slice(1) : tokens).join(" ");
  return PRONOUN_IMPERATIVE.test(body) || SELECTION_IMPERATIVE.test(body);
}
function extractFullIds(results) {
  if (!Array.isArray(results)) return [];
  return results.map((r) => r && typeof r.id === "string" ? r.id : "").filter((id) => id.length > 0);
}
function buildExcludeArgs(ids) {
  if (!ids || ids.length === 0) return [];
  return ["--exclude-ids", ...ids];
}
function unionCap(prior, fresh, cap) {
  const merged = [];
  const seen = /* @__PURE__ */ new Set();
  for (const id of [...prior || [], ...fresh || []]) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    merged.push(id);
  }
  if (merged.length <= cap) return merged;
  return merged.slice(merged.length - cap);
}
function readSurfacedIds(sessionId) {
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
    return parsed.filter((x) => typeof x === "string" && x.length > 0);
  } catch {
    return [];
  }
}
function persistSurfacedIds(sessionId, freshIds) {
  if (!sessionId || !freshIds || freshIds.length === 0) return;
  const capped = freshIds.slice(0, SURFACED_ID_CAP);
  const pythonCode = `
import asyncpg, json

db_url = os.environ['CONTINUOUS_CLAUDE_DB_URL']
session_id = sys.argv[1]
fresh = json.loads(sys.argv[2])

async def main():
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "UPDATE sessions SET surfaced_learning_ids = "
            "ARRAY(SELECT DISTINCT unnest("
            "COALESCE(surfaced_learning_ids, '{}'::uuid[]) || $2::uuid[]"
            ")) WHERE claude_session_id = $1",
            session_id,
            fresh,
        )
    finally:
        await conn.close()

asyncio.run(main())
`;
  try {
    runPgQuery(pythonCode, [sessionId, JSON.stringify(capped)]);
  } catch {
  }
}
function checkMemoryRelevance(intent, projectDir, excludeIds = []) {
  if (!intent || intent.length < 3) return null;
  const opcDir = getOpcDir();
  if (!opcDir) return null;
  const searchTerm = sanitizeSearchTerm(intent);
  if (!searchTerm) return null;
  const projectTag = projectDir ? projectDir.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "" : "";
  const safeProjectTag = projectTag && !projectTag.startsWith("-") ? projectTag : "";
  const tagArgs = safeProjectTag ? ["--tags", safeProjectTag] : [];
  const projectArgs = safeProjectTag ? ["--project", safeProjectTag, "--project-first"] : [];
  const result = spawnSync2("uv", [
    "run",
    "python",
    "scripts/core/recall_learnings.py",
    "--query",
    searchTerm,
    "--k",
    "3",
    "--json",
    "--provider",
    "voyage",
    "--source",
    "hook",
    ...tagArgs,
    ...projectArgs,
    // Issue #228 item 2: drop already-surfaced learnings BEFORE rank so the
    // hook stops re-surfacing the same top memories every turn. Appended to
    // the argv array (not shell) so full UUIDs are passed safely.
    ...buildExcludeArgs(excludeIds)
  ], {
    encoding: "utf-8",
    cwd: opcDir,
    env: {
      ...process.env,
      // Never rewrite opc's uv.lock from a hook-triggered uv run (issue #71).
      UV_FROZEN: "1",
      PYTHONPATH: opcDir
    },
    timeout: 5e3
    // 5s timeout for fast check
  });
  if (result.status !== 0 || !result.stdout) {
    return null;
  }
  try {
    const data = JSON.parse(result.stdout);
    if (!data.results || data.results.length === 0) {
      return null;
    }
    const results = data.results.slice(0, 3).map((r) => {
      const content = r.content || "";
      const preview = content.split("\n").filter((l) => l.trim().length > 0).map((l) => l.trim()).join(" ").slice(0, 120);
      return {
        id: (r.id || "unknown").slice(0, 8),
        type: r.learning_type || r.type || "UNKNOWN",
        content: preview + (content.length > 120 ? "..." : ""),
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
  const input = JSON.parse(readStdin());
  const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd;
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }
  if (input.prompt.length < 15) {
    return;
  }
  if (input.prompt.trim().startsWith("/")) {
    return;
  }
  if (isConversationalTurn(input.prompt)) {
    return;
  }
  const intent = extractIntent(input.prompt);
  if (intent.length < 3) {
    return;
  }
  const priorSurfaced = readSurfacedIds(input.session_id);
  const match = checkMemoryRelevance(intent, projectDir, priorSurfaced);
  if (match) {
    persistSurfacedIds(
      input.session_id,
      unionCap(priorSurfaced, match.fullIds, SURFACED_ID_CAP)
    );
    const resultLines = match.results.map(
      (r, i) => `${i + 1}. [${r.type}] ${r.content} (id: ${r.id})`
    ).join("\n");
    const claudeContext = `MEMORY MATCH (${match.count} results) for "${intent}":
${resultLines}
Use /recall "${intent}" for full content. Disclose if helpful.`;
    console.log(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: claudeContext
      }
    }));
  }
}
if (typeof process !== "undefined" && process.argv[1] && (process.argv[1].endsWith("memory-awareness.ts") || process.argv[1].endsWith("memory-awareness.js") || process.argv[1].endsWith("memory-awareness.mjs"))) {
  main().catch(() => {
  });
}
export {
  SURFACED_ID_CAP,
  buildExcludeArgs,
  extractFullIds,
  isConversationalTurn,
  persistSurfacedIds,
  readSurfacedIds,
  sanitizeSearchTerm,
  unionCap
};
