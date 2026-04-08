/*!
 * PHP Test Runner Hook (PostToolUse)
 *
 * Runs composer-defined test scripts after Edit/Write of PHP source files.
 * Detects available scripts from composer.json (test, phpcs, phpstan).
 * Skips vendor, node_modules, and non-PHP files.
 * Reports test results as additional context.
 */

import { readFileSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import * as path from 'path';

interface HookInput {
  tool_name: string;
  tool_input: {
    file_path?: string;
  };
  tool_response?: {
    filePath?: string;
    file_path?: string;
  };
}

interface HookOutput {
  hookSpecificOutput?: {
    hookEventName: string;
    additionalContext?: string;
  };
}

interface ComposerJson {
  scripts?: Record<string, string | string[]>;
}

function getComposerScripts(projectDir: string): Set<string> {
  const composerPath = path.join(projectDir, 'composer.json');
  if (!existsSync(composerPath)) return new Set();
  try {
    const composer: ComposerJson = JSON.parse(readFileSync(composerPath, 'utf-8'));
    return new Set(Object.keys(composer.scripts || {}));
  } catch {
    return new Set();
  }
}

function runCommand(cmd: string, cwd: string, timeout: number): { ok: boolean; output: string } {
  try {
    const result = execSync(cmd, {
      cwd,
      timeout,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { ok: true, output: result };
  } catch (err: unknown) {
    const execErr = err as { stdout?: string; stderr?: string };
    return { ok: false, output: (execErr.stdout || '') + (execErr.stderr || '') };
  }
}

function extractPhpunitSummary(output: string): string {
  const lines = output.trim().split('\n');

  // Look for the summary line: "OK (X tests, Y assertions)" or "Tests: X, ..."
  const summaryLine = lines.find(
    (l) => l.includes('OK (') || l.includes('FAILURES!') || l.includes('ERRORS!')
  );

  // Also grab the counts line
  const countsLine = lines.find(
    (l) => /Tests:\s+\d+/.test(l) || /\d+ tests?, \d+ assertions?/.test(l)
  );

  if (summaryLine?.includes('OK')) {
    return countsLine || summaryLine;
  }

  // Failure case — grab specific failure names
  const failures: string[] = [];
  const failedTests = lines.filter((l) => /^\d+\)\s/.test(l.trim()));
  for (const f of failedTests.slice(0, 5)) {
    failures.push(`  ${f.trim()}`);
  }

  const parts = [countsLine || summaryLine || 'Tests failed'];
  if (failures.length > 0) {
    parts.push(...failures);
  }
  return parts.join('\n');
}

function extractPhpcsSummary(output: string): string {
  const lines = output.trim().split('\n');

  // Look for summary: "FOUND X ERRORS AND Y WARNINGS"
  const foundLine = lines.find((l) => l.includes('FOUND'));
  if (foundLine) return foundLine.trim();

  // No issues
  if (output.includes('No violations')) return 'No violations';
  return lines[lines.length - 1]?.trim() || 'Check complete';
}

async function main() {
  const input: HookInput = JSON.parse(readFileSync(0, 'utf-8'));

  if (input.tool_name !== 'Edit' && input.tool_name !== 'Write') {
    console.log('{}');
    return;
  }

  const filePath =
    input.tool_input?.file_path ||
    input.tool_response?.filePath ||
    input.tool_response?.file_path;

  if (!filePath || typeof filePath !== 'string') {
    console.log('{}');
    return;
  }

  // Only PHP files
  if (!filePath.endsWith('.php')) {
    console.log('{}');
    return;
  }

  // Skip vendor, node_modules
  if (filePath.includes('/vendor/') || filePath.includes('/node_modules/')) {
    console.log('{}');
    return;
  }

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  // Must have a composer.json with scripts
  const scripts = getComposerScripts(projectDir);
  if (scripts.size === 0) {
    console.log('{}');
    return;
  }

  const results: string[] = [];

  // 1. Run phpcs if available (fast, code standards)
  if (scripts.has('phpcs')) {
    const phpcs = runCommand('composer phpcs -- --no-colors -q 2>&1', projectDir, 30_000);
    if (phpcs.ok) {
      results.push('phpcs: OK');
    } else {
      results.push(`phpcs: ${extractPhpcsSummary(phpcs.output)}`);
    }
  }

  // 2. Run phpunit/test if available (main test suite)
  if (scripts.has('test')) {
    const test = runCommand(
      'XDEBUG_MODE=off composer test -- --no-coverage --colors=never 2>&1',
      projectDir,
      120_000
    );
    if (test.ok) {
      results.push(`phpunit: ${extractPhpunitSummary(test.output)}`);
    } else {
      results.push(`phpunit: FAILED\n${extractPhpunitSummary(test.output)}`);
    }
  }

  if (results.length === 0) {
    console.log('{}');
    return;
  }

  const output: HookOutput = {
    hookSpecificOutput: {
      hookEventName: 'PostToolUse',
      additionalContext: results.join('\n'),
    },
  };
  console.log(JSON.stringify(output));
}

main().catch(() => console.log('{}'));
