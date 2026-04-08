// src/modern-cli-enforcer.ts
/*!
 * PreToolUse Hook: Modern CLI Enforcer
 *
 * Intercepts Bash tool calls that use legacy CLI commands and denies them
 * with a hint about the correct modern alternative.
 *
 * Catches commands that bypass settings.json deny rules (e.g., piped commands,
 * subshells, or commands settings.json patterns don't match).
 */
var REWRITABLE_COMMANDS = {
  "ls": ["eza", "-la", false],
  "du": ["dust", "", false],
  "df": ["duf", "", false],
  "top": ["btm", "", false],
  "htop": ["btm", "", false],
  "ps": ["procs", "", false],
  "pgrep": ["procs", "", false],
  "diff": ["delta", "", false],
  "python": ["uv", "run python", false],
  "python3": ["uv", "run python", false]
};
var DENY_COMMANDS = {
  "grep": ["Grep tool (or rg)", "ripgrep is faster and respects .gitignore", true],
  "egrep": ["Grep tool (or rg)", "ripgrep supports extended regex natively", true],
  "fgrep": ["Grep tool (or rg)", "ripgrep with --fixed-strings", true],
  "find": ["Glob tool (or fd)", "fd is faster and respects .gitignore", false],
  "cat": ["Read tool (or bat)", "bat provides syntax highlighting", false],
  "head": ["Read tool with limit", "Read tool supports offset and limit", true],
  "tail": ["Read tool with offset", "Read tool supports offset and limit", true],
  "less": ["Read tool", "Read tool provides file contents directly", false],
  "more": ["Read tool", "Read tool provides file contents directly", false],
  "sed": ["Edit tool", "Edit tool provides safe, targeted modifications", true],
  "awk": ["Edit tool", "Edit tool is safer for file modifications", true]
};
var REPLACEABLE_PIPE_SOURCES = {
  "fd": "Use the Glob tool with pattern matching instead of fd | head",
  "rg": "Use the Grep tool with head_limit parameter instead of rg | head",
  "eza": "Use the Glob tool instead of eza | head",
  "find": "Use the Glob tool with pattern matching instead of find | head",
  "ls": "Use the Glob tool instead of ls | head",
  "cat": "Use the Read tool with limit parameter instead of cat | head",
  "bat": "Use the Read tool with limit parameter instead of bat | head",
  "grep": "Use the Grep tool with head_limit parameter instead of grep | head"
};
function extractCommands(command) {
  const found = [];
  const chains = command.split(/\s*(?:\|\||&&|[;]|\$\()\s*/);
  for (const chain of chains) {
    const pipeSegments = chain.split(/\s*\|\s*/);
    let prevCmd;
    for (let i = 0; i < pipeSegments.length; i++) {
      const trimmed = pipeSegments[i].trim();
      if (!trimmed) continue;
      const cleaned = trimmed.replace(/^(?:\w+=\S+\s+)*/, "").replace(/^(?:sudo|command|env)\s+/, "").trim();
      const match = cleaned.match(/^([a-zA-Z0-9_.-]+)/);
      if (match) {
        found.push({ cmd: match[1], isPiped: i > 0, prevCmd });
        prevCmd = match[1];
      }
    }
  }
  return found;
}
function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
  });
}
async function main() {
  let input;
  try {
    input = JSON.parse(await readStdin());
  } catch {
    console.log("{}");
    return;
  }
  if (input.tool_name !== "Bash") {
    console.log("{}");
    return;
  }
  if (!input.tool_input || typeof input.tool_input.command !== "string") {
    console.log("{}");
    return;
  }
  const command = input.tool_input.command;
  const commands = extractCommands(command);
  const EZA_ALLOWED_FLAGS = /--tree|-T|--git/;
  const denyViolations = [];
  const rewrites = [];
  for (const { cmd, isPiped, prevCmd } of commands) {
    if (cmd === "eza") {
      const ezaMatch = command.match(/\beza\b(.+?)(?:\||&&|;|$)/);
      if (ezaMatch && !EZA_ALLOWED_FLAGS.test(ezaMatch[1])) {
        denyViolations.push({ cmd: "eza", replacement: "Glob tool for directory listing" });
      }
      continue;
    }
    const denyEntry = DENY_COMMANDS[cmd];
    if (denyEntry) {
      const [replacement, , pipeOk] = denyEntry;
      if (isPiped && pipeOk && prevCmd && REPLACEABLE_PIPE_SOURCES[prevCmd]) {
        denyViolations.push({ cmd: `${prevCmd} | ${cmd}`, replacement: REPLACEABLE_PIPE_SOURCES[prevCmd] });
        continue;
      }
      if (isPiped && pipeOk) continue;
      denyViolations.push({ cmd, replacement });
      continue;
    }
    const rewriteEntry = REWRITABLE_COMMANDS[cmd];
    if (rewriteEntry) {
      const [newCmd, defaultFlags, pipeOk] = rewriteEntry;
      if (isPiped && pipeOk) continue;
      rewrites.push({ from: cmd, to: newCmd, flags: defaultFlags });
    }
  }
  if (denyViolations.length === 0 && rewrites.length === 0) {
    console.log("{}");
    return;
  }
  if (denyViolations.length > 0) {
    const reason = denyViolations.map((v) => `${v.cmd} -> ${v.replacement}`).join(", ");
    const modelContext = denyViolations.map((v) => `${v.cmd} blocked. Use: ${v.replacement}`).join("; ");
    const output2 = {
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: reason,
        additionalContext: modelContext
      }
    };
    console.log(JSON.stringify(output2));
    return;
  }
  let rewrittenCommand = command;
  for (const { from, to, flags } of rewrites) {
    const pattern = new RegExp(`\\b${from}\\b`);
    const replacement = flags ? `${to} ${flags}` : to;
    rewrittenCommand = rewrittenCommand.replace(pattern, replacement);
  }
  const rewriteHints = rewrites.map((r) => `${r.from} -> ${r.to}`).join(", ");
  const output = {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "allow",
      updatedInput: { command: rewrittenCommand },
      additionalContext: `Command rewritten: ${rewriteHints}`
    }
  };
  console.log(JSON.stringify(output));
}
main().catch(console.error);
