/*!
 * PreToolUse Hook: Modern CLI Enforcer
 *
 * Intercepts Bash tool calls that use legacy CLI commands and denies them
 * with a hint about the correct modern alternative.
 *
 * Catches commands that bypass settings.json deny rules (e.g., piped commands,
 * subshells, or commands settings.json patterns don't match).
 */

interface BashInput {
  command: string;
  description?: string;
}

interface PreToolUseInput {
  tool_name: string;
  tool_input: BashInput;
  session_id?: string;
}

interface HookOutput {
  hookSpecificOutput?: {
    hookEventName: string;
    permissionDecision?: 'allow' | 'deny' | 'ask';
    permissionDecisionReason?: string;
    additionalContext?: string;
    updatedInput?: Record<string, unknown>;
  };
  systemMessage?: string;
}

// Commands that can be auto-rewritten (Bash → Bash swap)
// Each entry: [replacement binary, default flags to add, pipeOk]
const REWRITABLE_COMMANDS: Record<string, [string, string, boolean]> = {
  'ls':      ['eza', '-la', false],
  'du':      ['dust', '', false],
  'df':      ['duf', '', false],
  'top':     ['btm', '', false],
  'htop':    ['btm', '', false],
  'ps':      ['procs', '', false],
  'pgrep':   ['procs', '', false],
  'diff':    ['delta', '', false],
  'python':  ['uv', 'run python', false],
  'python3': ['uv', 'run python', false],
};

// Commands that must be denied (Bash → Claude tool redirect)
// Each entry: [replacement suggestion, explanation, pipeOk]
const DENY_COMMANDS: Record<string, [string, string, boolean]> = {
  'grep':    ['Grep tool (or rg)', 'ripgrep is faster and respects .gitignore', true],
  'egrep':   ['Grep tool (or rg)', 'ripgrep supports extended regex natively', true],
  'fgrep':   ['Grep tool (or rg)', 'ripgrep with --fixed-strings', true],
  'find':    ['Glob tool (or fd)', 'fd is faster and respects .gitignore', false],
  'cat':     ['Read tool (or bat)', 'bat provides syntax highlighting', false],
  'head':    ['Read tool with limit', 'Read tool supports offset and limit', true],
  'tail':    ['Read tool with offset', 'Read tool supports offset and limit', true],
  'less':    ['Read tool', 'Read tool provides file contents directly', false],
  'more':    ['Read tool', 'Read tool provides file contents directly', false],
  'sed':     ['Edit tool', 'Edit tool provides safe, targeted modifications', true],
  'awk':     ['Edit tool', 'Edit tool is safer for file modifications', true],
};

interface ParsedCommand {
  cmd: string;
  isPiped: boolean;  // true if this command appears after a pipe |
  prevCmd?: string;   // the command before the pipe (if piped)
}

// Commands whose piped output should use dedicated tools instead of head/tail/grep
// fd/rg/eza output can be replaced by Glob/Grep with head_limit
const REPLACEABLE_PIPE_SOURCES: Record<string, string> = {
  'fd':  'Use the Glob tool with pattern matching instead of fd | head',
  'rg':  'Use the Grep tool with head_limit parameter instead of rg | head',
  'eza': 'Use the Glob tool instead of eza | head',
  'find':'Use the Glob tool with pattern matching instead of find | head',
  'ls':  'Use the Glob tool instead of ls | head',
  'cat': 'Use the Read tool with limit parameter instead of cat | head',
  'bat': 'Use the Read tool with limit parameter instead of bat | head',
  'grep':'Use the Grep tool with head_limit parameter instead of grep | head',
};

/**
 * Extract the base command from a shell command string.
 * Handles: direct commands, env vars, sudo, pipes, subshells, etc.
 * Tracks whether each command is after a pipe (output filtering).
 */
function extractCommands(command: string): ParsedCommand[] {
  const found: ParsedCommand[] = [];

  // First split on statement separators (&&, ||, ;) — each is an independent command chain
  const chains = command.split(/\s*(?:\|\||&&|[;]|\$\()\s*/);

  for (const chain of chains) {
    // Within each chain, split on pipe | to track position
    const pipeSegments = chain.split(/\s*\|\s*/);

    let prevCmd: string | undefined;
    for (let i = 0; i < pipeSegments.length; i++) {
      const trimmed = pipeSegments[i].trim();
      if (!trimmed) continue;

      // Strip leading env var assignments (FOO=bar cmd), sudo, command, env, etc.
      const cleaned = trimmed
        .replace(/^(?:\w+=\S+\s+)*/, '')       // env vars
        .replace(/^(?:sudo|command|env)\s+/, '') // prefixes
        .trim();

      // First word is the command
      const match = cleaned.match(/^([a-zA-Z0-9_.-]+)/);
      if (match) {
        found.push({ cmd: match[1], isPiped: i > 0, prevCmd });
        prevCmd = match[1];
      }
    }
  }

  return found;
}

function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
  });
}

async function main() {
  let input: PreToolUseInput;
  try {
    input = JSON.parse(await readStdin());
  } catch {
    console.log('{}');
    return;
  }

  // Only intercept Bash tool
  if (input.tool_name !== 'Bash') {
    console.log('{}');
    return;
  }

  if (!input.tool_input || typeof input.tool_input.command !== 'string') {
    console.log('{}');
    return;
  }

  const command = input.tool_input.command;
  const commands = extractCommands(command);

  // Collect deny violations and rewrite candidates separately
  const EZA_ALLOWED_FLAGS = /--tree|-T|--git/;
  const denyViolations: Array<{ cmd: string; replacement: string }> = [];
  const rewrites: Array<{ from: string; to: string; flags: string }> = [];

  for (const { cmd, isPiped, prevCmd } of commands) {
    // Check eza: allow --tree/--git, deny plain usage
    if (cmd === 'eza') {
      const ezaMatch = command.match(/\beza\b(.+?)(?:\||&&|;|$)/);
      if (ezaMatch && !EZA_ALLOWED_FLAGS.test(ezaMatch[1])) {
        denyViolations.push({ cmd: 'eza', replacement: 'Glob tool for directory listing' });
      }
      continue;
    }

    // Check deny-only commands first
    const denyEntry = DENY_COMMANDS[cmd];
    if (denyEntry) {
      const [replacement, , pipeOk] = denyEntry;

      // Special case: piped after replaceable source
      if (isPiped && pipeOk && prevCmd && REPLACEABLE_PIPE_SOURCES[prevCmd]) {
        denyViolations.push({ cmd: `${prevCmd} | ${cmd}`, replacement: REPLACEABLE_PIPE_SOURCES[prevCmd] });
        continue;
      }

      if (isPiped && pipeOk) continue;
      denyViolations.push({ cmd, replacement });
      continue;
    }

    // Check rewritable commands
    const rewriteEntry = REWRITABLE_COMMANDS[cmd];
    if (rewriteEntry) {
      const [newCmd, defaultFlags, pipeOk] = rewriteEntry;
      if (isPiped && pipeOk) continue;
      rewrites.push({ from: cmd, to: newCmd, flags: defaultFlags });
    }
  }

  // Nothing to do
  if (denyViolations.length === 0 && rewrites.length === 0) {
    console.log('{}');
    return;
  }

  // If any deny violations exist, deny the whole command
  if (denyViolations.length > 0) {
    const reason = denyViolations.map(v => `${v.cmd} -> ${v.replacement}`).join(', ');
    const modelContext = denyViolations.map(v => `${v.cmd} blocked. Use: ${v.replacement}`).join('; ');

    const output: HookOutput = {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        permissionDecision: 'deny',
        permissionDecisionReason: reason,
        additionalContext: modelContext,
      },
    };
    console.log(JSON.stringify(output));
    return;
  }

  // All violations are rewritable — transform the command
  let rewrittenCommand = command;
  for (const { from, to, flags } of rewrites) {
    // Replace the command, preserving its arguments
    // Match word boundary to avoid partial replacements
    const pattern = new RegExp(`\\b${from}\\b`);
    const replacement = flags ? `${to} ${flags}` : to;
    rewrittenCommand = rewrittenCommand.replace(pattern, replacement);
  }

  const rewriteHints = rewrites.map(r => `${r.from} -> ${r.to}`).join(', ');
  const output: HookOutput = {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: 'allow',
      updatedInput: { command: rewrittenCommand },
      additionalContext: `Command rewritten: ${rewriteHints}`,
    },
  };
  console.log(JSON.stringify(output));
}

main().catch(console.error);
