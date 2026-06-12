/**
 * PostToolUse Hook — keeps the coordination layer's `sessions.working_on`
 * current so peer sessions can see what each session is doing.
 *
 * Fires on TodoWrite | TaskCreate | TaskUpdate. The challenge: a
 * `TaskUpdate` payload carries only `{taskId, status}` with no label, and
 * native task state lives in-process (no on-disk store a hook can read).
 * So we keep a small session-scoped `taskId -> label` cache, populated from
 * `TaskCreate` (whose tool_response carries the assigned id), and resolve
 * the label when a later `TaskUpdate` marks that id `in_progress`.
 *
 * `TodoWrite` is self-contained (the payload has the full list with
 * statuses) and needs no cache.
 *
 * Always outputs { result: "continue" } — never blocks tool execution.
 * The DB write is detached fire-and-forget (no latency on the tool call).
 */

import { readFileSync, writeFileSync, renameSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { updateWorkingOnDetached, isValidId } from './shared/db-utils-pg.js';
import { getProject } from './shared/session-id.js';

interface TodoItem {
  content?: string;
  activeForm?: string;
  status?: string;
}

interface PostToolUseInput {
  session_id?: string;
  tool_name?: string;
  tool_input?: {
    todos?: TodoItem[];
    subject?: string;
    activeForm?: string;
    taskId?: string;
    status?: string;
  };
  tool_response?: unknown;
}

interface WorkingOnCache {
  tasks: Record<string, string>;
  currentId: string | null;
}

const EMPTY_CACHE: WorkingOnCache = { tasks: {}, currentId: null };

// --- pure core -------------------------------------------------------------

/** Extract the task id GitHub-style task tools assign in their response text. */
export function parseCreatedTaskId(toolResponse: unknown): string | null {
  const text =
    typeof toolResponse === 'string'
      ? toolResponse
      : toolResponse && typeof toolResponse === 'object'
        ? JSON.stringify(toolResponse)
        : '';
  const m = text.match(/Task #(\d+)\b/);
  return m ? m[1] : null;
}

/** Label of the first in-progress todo, or '' when none is in progress. */
export function pickTodoInProgress(todos: TodoItem[] | undefined): string {
  const t = (todos || []).find((x) => x.status === 'in_progress');
  if (!t) return '';
  return (t.activeForm || t.content || '').trim();
}

/**
 * Decide the new working_on value (or null = no DB write) and the next cache
 * state. Pure: no I/O, no clock. `workingOn` of '' means "clear it".
 */
export function deriveWorkingOn(
  input: PostToolUseInput,
  cache: WorkingOnCache,
): { workingOn: string | null; cache: WorkingOnCache } {
  const tool = input.tool_name;
  const ti = input.tool_input || {};
  const next: WorkingOnCache = {
    tasks: { ...cache.tasks },
    currentId: cache.currentId,
  };

  if (tool === 'TodoWrite') {
    return { workingOn: pickTodoInProgress(ti.todos), cache: next };
  }

  if (tool === 'TaskCreate') {
    const id = parseCreatedTaskId(input.tool_response);
    const label = (ti.activeForm || ti.subject || '').trim();
    if (id && label) next.tasks[id] = label;
    return { workingOn: null, cache: next }; // task is pending; don't write
  }

  if (tool === 'TaskUpdate') {
    const id = ti.taskId;
    if (!id) return { workingOn: null, cache: next };
    if (ti.status === 'in_progress') {
      const label = next.tasks[id];
      if (!label) return { workingOn: null, cache: next };
      next.currentId = id;
      return { workingOn: label, cache: next };
    }
    if (ti.status === 'completed' || ti.status === 'deleted') {
      // Drop the finished task's label so the cache stays bounded to the
      // currently-active tasks rather than growing for the whole session.
      delete next.tasks[id];
      if (id === next.currentId) {
        next.currentId = null;
        return { workingOn: '', cache: next }; // active task finished — clear
      }
      return { workingOn: null, cache: next };
    }
  }

  return { workingOn: null, cache: next };
}

// --- I/O edges -------------------------------------------------------------

function cachePath(sessionId: string): string {
  return join(
    process.env.HOME || process.env.USERPROFILE || '',
    '.claude',
    'cache',
    'working-on',
    `${sessionId}.json`,
  );
}

function readCache(sessionId: string): WorkingOnCache {
  try {
    const raw = readFileSync(cachePath(sessionId), 'utf-8');
    const parsed = JSON.parse(raw) as Partial<WorkingOnCache>;
    return {
      tasks: parsed.tasks && typeof parsed.tasks === 'object' ? parsed.tasks : {},
      currentId: typeof parsed.currentId === 'string' ? parsed.currentId : null,
    };
  } catch {
    return { ...EMPTY_CACHE, tasks: {} };
  }
}

function writeCache(sessionId: string, cache: WorkingOnCache): void {
  const p = cachePath(sessionId);
  try {
    const dir = join(p, '..');
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    const tmp = `${p}.tmp`;
    writeFileSync(tmp, JSON.stringify(cache), 'utf-8');
    renameSync(tmp, p);
  } catch {
    // Cache is best-effort; a write failure must never break the hook.
  }
}

export function main(): void {
  let input: PostToolUseInput;
  try {
    input = JSON.parse(readFileSync(0, 'utf-8')) as PostToolUseInput;
  } catch {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const sessionId = input.session_id;
  const relevant =
    input.tool_name === 'TodoWrite' ||
    input.tool_name === 'TaskCreate' ||
    input.tool_name === 'TaskUpdate';
  if (!sessionId || !isValidId(sessionId) || !relevant) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const cache = readCache(sessionId);
  const { workingOn, cache: nextCache } = deriveWorkingOn(input, cache);
  writeCache(sessionId, nextCache);

  if (workingOn !== null) {
    try {
      updateWorkingOnDetached(sessionId, getProject(), workingOn);
    } catch {
      // Best-effort: a missing DB URL or spawn failure must never break the
      // hook. PostToolUse must always emit continue (issue #65 review r1).
    }
  }

  console.log(JSON.stringify({ result: 'continue' }));
}

// Run if executed directly
if (
  typeof process !== 'undefined' &&
  process.argv[1] &&
  (process.argv[1].endsWith('working-on-sync.ts') ||
    process.argv[1].endsWith('working-on-sync.js') ||
    process.argv[1].endsWith('working-on-sync.mjs'))
) {
  main();
}
