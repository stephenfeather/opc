/*!
 * Failure Detection Hook (PostToolUse:Bash|Task)
 *
 * Detects errors in Bash and Task tool responses and suggests
 * documentation searches for resolution.
 *
 * - Bash: checks exit_code !== 0 and extracts error from stderr/stdout
 * - Task: scans response text for error/exception/crash patterns
 * - Extracts specific Python exception context (ModuleNotFoundError, etc.)
 * - Suggests Nia documentation search with the error context
 */

import { readFileSync } from 'fs';

interface BashResponse {
  exit_code: number;
  stdout?: string;
  stderr?: string;
}

interface PostToolUseInput {
  tool_name: string;
  tool_input: {
    command?: string;
    [key: string]: unknown;
  };
  tool_response: unknown;
}

interface FailureResult {
  failed: boolean;
  errorText: string;
}

interface HookOutput {
  result: 'continue';
  message?: string;
}

const TASK_ERROR_PATTERNS: RegExp[] = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bexception\b/i,
  /\bcrash(ed)?\b/i,
  /\btimeout\b/i,
  /\babort(ed)?\b/i,
  /\bpanic\b/i,
  /\bfatal\b/i,
];

const ERROR_CONTEXT_PATTERNS: RegExp[] = [
  /ModuleNotFoundError:\s*No module named\s*['"]?(\w+)['"]?/i,
  /ImportError:\s*(.+)/i,
  /TypeError:\s*(.+)/i,
  /ValueError:\s*(.+)/i,
  /AttributeError:\s*(.+)/i,
  /NameError:\s*(.+)/i,
  /SyntaxError:\s*(.+)/i,
  /RuntimeError:\s*(.+)/i,
  /KeyError:\s*(.+)/i,
  /FileNotFoundError:\s*(.+)/i,
  /PermissionError:\s*(.+)/i,
  /ConnectionError:\s*(.+)/i,
  /OSError:\s*(.+)/i,
  /Error:\s*(.+)/i,
  /error:\s*(.+)/i,
  /failed:\s*(.+)/i,
  /FAILED:\s*(.+)/i,
];

function isBashFailure(response: unknown): FailureResult {
  if (typeof response === 'object' && response !== null) {
    const bashResponse = response as BashResponse;
    if (typeof bashResponse.exit_code === 'number' && bashResponse.exit_code !== 0) {
      const stderr = bashResponse.stderr || '';
      const stdout = bashResponse.stdout || '';
      return { failed: true, errorText: stderr || stdout };
    }
  }
  return { failed: false, errorText: '' };
}

function isTaskFailure(response: unknown): FailureResult {
  let text = '';
  if (typeof response === 'string') {
    text = response;
  } else if (typeof response === 'object' && response !== null) {
    text = JSON.stringify(response);
  }

  for (const pattern of TASK_ERROR_PATTERNS) {
    if (pattern.test(text)) {
      return { failed: true, errorText: text };
    }
  }
  return { failed: false, errorText: '' };
}

function extractErrorContext(
  errorText: string,
  toolInput: PostToolUseInput['tool_input'],
): string {
  for (const pattern of ERROR_CONTEXT_PATTERNS) {
    const match = pattern.exec(errorText);
    if (match) {
      const context = match[1] || match[0];
      return context.substring(0, 100).trim();
    }
  }

  const firstLine = errorText.split('\n')[0] || '';
  if (firstLine.length > 100) {
    return firstLine.substring(0, 100).trim();
  }

  if (toolInput.command && typeof toolInput.command === 'string') {
    return `command failed: ${toolInput.command.substring(0, 50)}`;
  }

  return 'execution failed';
}

function buildNiaSearchCommand(errorContext: string): string {
  const escapedContext = errorContext.replace(/'/g, "'\\''").replace(/"/g, '\\"');
  return `uv run python -m runtime.harness scripts/nia_docs.py search universal "${escapedContext}" --limit 5`;
}

async function main(): Promise<void> {
  let input: PostToolUseInput;
  try {
    const rawInput = readFileSync(0, 'utf-8');
    input = JSON.parse(rawInput);
  } catch {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  if (input.tool_name !== 'Bash' && input.tool_name !== 'Task') {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  let failed = false;
  let errorText = '';

  if (input.tool_name === 'Bash') {
    const result = isBashFailure(input.tool_response);
    failed = result.failed;
    errorText = result.errorText;
  } else if (input.tool_name === 'Task') {
    const result = isTaskFailure(input.tool_response);
    failed = result.failed;
    errorText = result.errorText;
  }

  if (!failed) {
    console.log(JSON.stringify({ result: 'continue' }));
    return;
  }

  const errorContext = extractErrorContext(errorText, input.tool_input);
  const niaCommand = buildNiaSearchCommand(errorContext);

  const output: HookOutput = {
    result: 'continue',
    message: `
---
**Build/Execution Failure Detected**

Consider searching documentation for help:
\`\`\`bash
${niaCommand}
\`\`\`

Error context: ${errorContext.substring(0, 200)}
---`,
  };
  console.log(JSON.stringify(output));
}

main().catch(() => {
  console.log(JSON.stringify({ result: 'continue' }));
});
