// src/bash-error-pause.ts
import { readFileSync } from "fs";
//! @hook PostToolUse:Bash @preserve
var WARNING_PATTERNS = [
  /\bwarn(ing)?\b/i,
  /\bdeprecated\b/i,
  /\bWARN\b/
];
var ERROR_PATTERNS = [
  /\berror\b/i,
  /\bfailed\b/i,
  /\bfailure\b/i,
  /\bexception\b/i,
  /\bfatal\b/i,
  /\bpanic\b/i,
  /\bsegfault\b/i,
  /\bsegmentation fault\b/i,
  /\baborted\b/i,
  /\btraceback\b/i,
  /\bERROR\b/,
  /\bFAILED\b/,
  /\bFATAL\b/,
  /exit code [1-9]\d*/i,
  /returned? [1-9]\d*/i
];
var FALSE_POSITIVE_PATTERNS = [
  /0 errors?\b/i,
  /no errors?\b/i,
  /error[_-]?handl/i,
  /error[_-]?messag/i,
  /error[_-]?code/i,
  /error[_-]?type/i,
  /error[_-]?class/i,
  /on_?error/i,
  /if.*error/i,
  /catch.*error/i,
  /throw.*error/i,
  /console\.(warn|error)/i,
  /stderr/i,
  /\bwarning:\s*0\b/i,
  /0 warning/i,
  /no warning/i
];
function extractResponseText(response) {
  if (typeof response === "string") return response;
  if (response && typeof response === "object") {
    const resp = response;
    const parts = [];
    if (typeof resp.stdout === "string") parts.push(resp.stdout);
    if (typeof resp.stderr === "string") parts.push(resp.stderr);
    if (parts.length > 0) return parts.join("\n");
    return JSON.stringify(response);
  }
  return String(response ?? "");
}
function hasNonFalsePositiveMatch(text, patterns) {
  for (const pattern of patterns) {
    const match = pattern.exec(text);
    if (!match) continue;
    const lineStart = text.lastIndexOf("\n", match.index) + 1;
    const lineEnd = text.indexOf("\n", match.index);
    const line = text.slice(lineStart, lineEnd === -1 ? void 0 : lineEnd);
    const isFalsePositive = FALSE_POSITIVE_PATTERNS.some((fp) => fp.test(line));
    if (!isFalsePositive) return true;
  }
  return false;
}
function main() {
  let input;
  try {
    const stdinContent = readFileSync(0, "utf-8");
    input = JSON.parse(stdinContent);
  } catch {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  if (input.tool_name !== "Bash") {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const text = extractResponseText(input.tool_response);
  if (!text.trim()) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const hasError = hasNonFalsePositiveMatch(text, ERROR_PATTERNS);
  const hasWarning = !hasError && hasNonFalsePositiveMatch(text, WARNING_PATTERNS);
  if (!hasError && !hasWarning) {
    console.log(JSON.stringify({ result: "continue" }));
    return;
  }
  const severity = hasError ? "ERROR" : "WARNING";
  const output = {
    result: "continue",
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: `STOP: ${severity} detected in Bash output. Verify the cause before explaining it to the user. Do NOT guess \u2014 read the error, check assumptions, trace the root cause.`
    }
  };
  console.log(JSON.stringify(output));
}
main();
export {
  main
};
