// src/timestamp-inject.ts
import { readFileSync } from "fs";
//! @hook UserPromptSubmit @preserve
function readStdin() {
  return readFileSync(0, "utf-8");
}
function formatTimestamp(now) {
  const iso = now.toISOString();
  const local = now.toLocaleString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
  return `Current time: ${local} (${tz}) | ISO: ${iso}`;
}
function main() {
  let input;
  try {
    input = JSON.parse(readStdin());
  } catch {
    return;
  }
  if (!input || typeof input !== "object" || !input.hook_event_name) {
    return;
  }
  if (process.env.CLAUDE_AGENT_ID) {
    return;
  }
  const now = /* @__PURE__ */ new Date();
  const timestamp = formatTimestamp(now);
  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: timestamp
    }
  }));
}
main();
