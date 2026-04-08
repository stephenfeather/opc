/*!
 * Session Start Memory Push Hook (SessionStart, type=startup)
 *
 * Proactively surfaces high-value, never-recalled learnings at session start.
 * Targets two pools:
 *   1. Stale high-confidence learnings for the current project
 *   2. Pattern representatives from anti_pattern / problem_solution clusters
 *
 * Calls push_learnings.py via subprocess (mirrors memory-awareness.ts pattern).
 * Injects results via hookSpecificOutput.additionalContext.
 */

import { readFileSync, existsSync } from 'fs';
import { spawnSync } from 'child_process';
import { join } from 'path';
import { getOpcDir } from './shared/opc-path.js';

interface SessionStartInput {
  session_id: string;
  type?: string;   // 'startup' | 'resume' | 'compact' | 'clear'
  source?: string;  // Same as type, per docs (check both)
}

interface PushResult {
  id: string;
  content: string;
  learning_type: string;
  confidence: string;
  pattern_label: string | null;
}

interface PushOutput {
  results: PushResult[];
  project?: string;
}

function main(): void {
  let input: SessionStartInput;
  try {
    const stdinContent = readFileSync(0, 'utf-8');
    input = JSON.parse(stdinContent) as SessionStartInput;
  } catch {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Only fire on startup (not resume/compact/clear)
  const eventType = input.type || input.source || 'startup';
  if (eventType !== 'startup') {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Skip for subagents
  if (process.env.CLAUDE_AGENT_ID) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Skip for daemon extraction sessions
  if (process.env.CLAUDE_MEMORY_EXTRACTION) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Check opt-out sentinel
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const sentinel = join(projectDir, '.claude', 'no-memory-push');
  if (existsSync(sentinel)) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const opcDir = getOpcDir();
  if (!opcDir) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Derive project name from CLAUDE_PROJECT_DIR
  const projectName = projectDir
    .replace(/[\\/]+$/, '')
    .split(/[\\/]/)
    .pop() ?? '';

  if (!projectName || projectName.startsWith('-')) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Call push_learnings.py
  const result = spawnSync('uv', [
    'run', 'python', 'scripts/core/push_learnings.py',
    '--project', projectName,
    '--k', '5',
    '--json',
    '--max-chars', '150'
  ], {
    encoding: 'utf-8',
    cwd: opcDir,
    env: {
      ...process.env,
      PYTHONPATH: opcDir
    },
    timeout: 8000
  });

  if (result.status !== 0 || !result.stdout) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  let data: PushOutput;
  try {
    data = JSON.parse(result.stdout) as PushOutput;
  } catch {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  if (!data.results || data.results.length === 0) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  // Format for Claude's context
  const resultLines = data.results.map((r, i) => {
    const base = `${i + 1}. [${r.learning_type}|${r.confidence}] `
      + `${r.content} (id: ${r.id})`;
    const label = r.pattern_label
      ? `\n   ↳ Pattern: "${r.pattern_label}"`
      : '';
    return base + label;
  }).join('\n');

  const context = [
    `PROACTIVE MEMORY (${data.results.length} learnings for "${projectName}"):`,
    resultLines,
    'These were surfaced proactively. Use /recall for full content.',
    'If any learning helps or misleads you, submit feedback: mcp__opc-memory__store_feedback(learning_id="<id>", helpful=true/false)',
  ].join('\n');

  console.log(JSON.stringify({
    result: 'continue',
    hookSpecificOutput: {
      hookEventName: 'SessionStart',
      additionalContext: context
    }
  }));
}

main();
