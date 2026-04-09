// src/session-id-inject.ts
import { readFileSync } from "fs";
/*!
 * SessionID Injection Hook (SessionStart)
 *
 * Injects the current session ID into every prompt as additionalContext,
 * enabling session-aware capabilities
 * in your hooks. This is especially useful for tracking and debugging
 * conversations across multiple interactions.
 */
function readStdin() {
  return readFileSync(0, "utf-8");
}
function main() {
  let input;
  try {
    input = JSON.parse(readStdin());
  } catch {
    return;
  }
  if (!input || typeof input !== "object" || !input.session_id) {
    return;
  }
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }
  let SessionId = `SessionId: ${input.session_id}`;
  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: SessionId
    }
  }));
}
main();
