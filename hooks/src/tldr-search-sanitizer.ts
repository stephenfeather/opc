/*!
 * PreToolUse Hook: TLDR Search Sanitizer
 *
 * Removes unsupported "--project" flag from `tldr search` Bash commands.
 * This prevents invalid argument errors like:
 *   tldr search "pattern" /path --project /path
 */

import { readFileSync } from 'fs';

interface PreToolUseInput {
  tool_name: string;
  tool_input: {
    command?: string;
    description?: string;
  };
}

interface HookOutput {
  hookSpecificOutput?: {
    hookEventName: string;
    permissionDecision: 'allow' | 'deny' | 'ask';
    permissionDecisionReason?: string;
    updatedInput?: Record<string, unknown>;
  };
  systemMessage?: string;
}

function tokenizeCommand(command: string): string[] {
  const tokens: string[] = [];
  const regex = /"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)'|`[^`]*`|\\\S+|\S+/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(command)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}

function isTldrSearch(tokens: string[]): boolean {
  if (tokens.length < 2) return false;
  if (tokens[0] === 'tldr' && tokens[1] === 'search') return true;
  if (tokens[0] === 'uv' && tokens[1] === 'run' && tokens[2] === 'tldr' && tokens[3] === 'search') {
    return true;
  }
  return false;
}

function sanitizeTldrSearch(command: string): { changed: boolean; sanitized: string } {
  const tokens = tokenizeCommand(command);
  if (!isTldrSearch(tokens)) {
    return { changed: false, sanitized: command };
  }

  const sanitizedTokens: string[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token === '--project') {
      // Skip this token and the following arg (if present).
      i += 1;
      continue;
    }
    if (token.startsWith('--project=')) {
      continue;
    }
    sanitizedTokens.push(token);
  }

  const sanitized = sanitizedTokens.join(' ');
  const changed = sanitized !== command;
  return { changed, sanitized };
}

async function main() {
  let input: PreToolUseInput;
  try {
    input = JSON.parse(readFileSync(0, 'utf-8')) as PreToolUseInput;
  } catch {
    console.log('{}');
    return;
  }

  if (input.tool_name !== 'Bash') {
    console.log('{}');
    return;
  }

  const command = input.tool_input?.command;
  if (!command || typeof command !== 'string') {
    console.log('{}');
    return;
  }

  const { changed, sanitized } = sanitizeTldrSearch(command);
  if (!changed) {
    console.log('{}');
    return;
  }

  const output: HookOutput = {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'allow',
      updatedInput: {
        ...input.tool_input,
        command: sanitized,
      }
    },
    systemMessage: '⚠️ Removed unsupported `--project` from `tldr search` command.',
  };

  console.log(JSON.stringify(output));
}

main().catch(() => {
  console.log('{}');
});
