// src/tldr-search-sanitizer.ts
import { readFileSync } from "fs";
/*!
 * PreToolUse Hook: TLDR Search Sanitizer
 *
 * Removes unsupported "--project" flag from `tldr search` Bash commands.
 * This prevents invalid argument errors like:
 *   tldr search "pattern" /path --project /path
 */
function tokenizeCommand(command) {
  const tokens = [];
  const regex = /"([^"\\]*(\\.[^"\\]*)*)"|'([^'\\]*(\\.[^'\\]*)*)'|`[^`]*`|\\\S+|\S+/g;
  let match;
  while ((match = regex.exec(command)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}
function isTldrSearch(tokens) {
  if (tokens.length < 2) return false;
  if (tokens[0] === "tldr" && tokens[1] === "search") return true;
  if (tokens[0] === "uv" && tokens[1] === "run" && tokens[2] === "tldr" && tokens[3] === "search") {
    return true;
  }
  return false;
}
function sanitizeTldrSearch(command) {
  const tokens = tokenizeCommand(command);
  if (!isTldrSearch(tokens)) {
    return { changed: false, sanitized: command };
  }
  const sanitizedTokens = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token === "--project") {
      i += 1;
      continue;
    }
    if (token.startsWith("--project=")) {
      continue;
    }
    sanitizedTokens.push(token);
  }
  const sanitized = sanitizedTokens.join(" ");
  const changed = sanitized !== command;
  return { changed, sanitized };
}
async function main() {
  let input;
  try {
    input = JSON.parse(readFileSync(0, "utf-8"));
  } catch {
    console.log("{}");
    return;
  }
  if (input.tool_name !== "Bash") {
    console.log("{}");
    return;
  }
  const command = input.tool_input?.command;
  if (!command || typeof command !== "string") {
    console.log("{}");
    return;
  }
  const { changed, sanitized } = sanitizeTldrSearch(command);
  if (!changed) {
    console.log("{}");
    return;
  }
  const output = {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "allow",
      updatedInput: {
        ...input.tool_input,
        command: sanitized
      }
    },
    systemMessage: "\u26A0\uFE0F Removed unsupported `--project` from `tldr search` command."
  };
  console.log(JSON.stringify(output));
}
main().catch(() => {
  console.log("{}");
});
