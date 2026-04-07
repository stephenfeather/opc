/**
 * Pytest Runner Hook (PostToolUse)
 *
 * Runs pytest after Edit/Write of Python source files.
 * Skips venv, .venv, vendor, node_modules, and non-Python files.
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

function hasPytestConfig(projectDir: string): boolean {
  return (
    existsSync(path.join(projectDir, 'pytest.ini')) ||
    existsSync(path.join(projectDir, 'pyproject.toml')) ||
    existsSync(path.join(projectDir, 'setup.cfg')) ||
    existsSync(path.join(projectDir, 'conftest.py')) ||
    existsSync(path.join(projectDir, 'tests', 'conftest.py'))
  );
}

function hasUv(): boolean {
  try {
    execSync('command -v uv', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
    return true;
  } catch {
    return false;
  }
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

  // Only Python files
  if (!filePath.endsWith('.py') && !filePath.endsWith('.pyx') && !filePath.endsWith('.pyi')) {
    console.log('{}');
    return;
  }

  // Skip virtual environments and vendor dirs
  if (
    filePath.includes('/venv/') ||
    filePath.includes('/.venv/') ||
    filePath.includes('/vendor/') ||
    filePath.includes('/node_modules/') ||
    filePath.includes('/__pycache__/')
  ) {
    console.log('{}');
    return;
  }

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();

  if (!hasPytestConfig(projectDir)) {
    console.log('{}');
    return;
  }

  const pytestCmd = hasUv() ? 'uv run pytest' : 'pytest';

  try {
    const result = execSync(`${pytestCmd} --tb=short -q --no-header 2>&1`, {
      cwd: projectDir,
      timeout: 120_000,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Tests passed — extract summary
    const lines = result.trim().split('\n');
    const summaryLine = lines.find(
      (l) => l.includes(' passed') || l.includes('no tests ran')
    );

    const output: HookOutput = {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse',
        additionalContext: `pytest: ${summaryLine || 'All tests passed'}`,
      },
    };
    console.log(JSON.stringify(output));
  } catch (err: unknown) {
    const execErr = err as { stdout?: string; stderr?: string };
    const combined = (execErr.stdout || '') + (execErr.stderr || '');
    const outputLines = combined.trim().split('\n');

    const failLines: string[] = [];
    failLines.push('pytest: TESTS FAILED');
    failLines.push('');

    // Grab the short summary line
    const summaryLine = outputLines.find(
      (l) => l.includes(' failed') || l.includes(' error')
    );
    if (summaryLine) {
      failLines.push(summaryLine.trim());
    }

    // Grab FAILED lines (specific test names)
    const failedTests = outputLines.filter((l) => l.startsWith('FAILED '));
    for (const test of failedTests.slice(0, 5)) {
      failLines.push(`  ${test.trim()}`);
    }

    // Grab short traceback snippets
    const tbLines = outputLines.filter(
      (l) =>
        l.includes('AssertionError') ||
        l.includes('Error:') ||
        l.includes('assert ')
    );
    for (const tb of tbLines.slice(0, 3)) {
      failLines.push(`  ${tb.trim()}`);
    }

    const output: HookOutput = {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse',
        additionalContext: failLines.join('\n'),
      },
    };
    console.log(JSON.stringify(output));
  }
}

main().catch(() => console.log('{}'));
