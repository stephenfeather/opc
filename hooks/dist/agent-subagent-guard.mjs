// src/agent-subagent-guard.ts
/*!
 * PreToolUse:Agent hook — blocks Agent calls with missing/null subagent_type.
 *
 * The general-purpose default (Claude Code internal) grants all tools including
 * Write/Edit, which is too permissive. This hook denies the call and suggests
 * appropriate specialist agent types based on the prompt content.
 */
var AGENT_SUGGESTIONS = [
  { keyword: /\b(fix|bug|debug|error|fail|broken|trace|root.?cause)\b/i, agent: "sleuth", description: "Bug investigation and root cause analysis" },
  { keyword: /\b(implement|build|create|add|feature|write code)\b/i, agent: "kraken", description: "Implementation (large) or spark (small fix)" },
  { keyword: /\b(quick fix|patch|tweak|minor|small change)\b/i, agent: "spark", description: "Lightweight fixes and quick tweaks" },
  { keyword: /\b(test|unit test|integration test|spec)\b/i, agent: "arbiter", description: "Unit and integration test execution" },
  { keyword: /\b(e2e|end.to.end|acceptance)\b/i, agent: "atlas", description: "End-to-end and acceptance tests" },
  { keyword: /\b(security|vulnerabilit|audit|cve|owasp)\b/i, agent: "aegis", description: "Security vulnerability analysis" },
  { keyword: /\b(refactor|migrat|restructur|rewrite)\b/i, agent: "phoenix", description: "Refactoring and migration planning" },
  { keyword: /\b(plan|design|architect|strateg)\b/i, agent: "architect", description: "Feature planning and design" },
  { keyword: /\b(review|code review|check quality)\b/i, agent: "critic", description: "Code review" },
  { keyword: /\b(document|readme|guide|explain)\b/i, agent: "scribe", description: "Documentation" },
  { keyword: /\b(research|find|search|explore|codebase)\b/i, agent: "scout", description: "Codebase exploration and pattern finding" },
  { keyword: /\b(external|docs|web|api|library|best practice)\b/i, agent: "oracle", description: "External research \u2014 web, docs, APIs" },
  { keyword: /\b(perform|profil|bottleneck|slow|memory|race)\b/i, agent: "profiler", description: "Performance profiling" },
  { keyword: /\b(release|version|changelog|deploy)\b/i, agent: "herald", description: "Release prep and changelog" }
];
var DEFAULT_SUGGESTIONS = [
  "  - scout: Codebase exploration and pattern finding",
  "  - kraken: Implementation (TDD workflow)",
  "  - spark: Lightweight fixes and quick tweaks"
];
function suggestAgents(prompt) {
  const matches = [];
  for (const { keyword, agent, description } of AGENT_SUGGESTIONS) {
    if (keyword.test(prompt)) {
      matches.push(`  - ${agent}: ${description}`);
    }
  }
  return matches.length > 0 ? matches.slice(0, 3) : DEFAULT_SUGGESTIONS;
}
var BLOCKED_TYPES = /* @__PURE__ */ new Set(["general-purpose"]);
function isSubagentTypeValid(subagentType) {
  if (typeof subagentType !== "string" || subagentType.trim().length === 0) {
    return false;
  }
  return !BLOCKED_TYPES.has(subagentType.trim().toLowerCase());
}
function buildDenyResponse(promptText) {
  const suggestions = suggestAgents(promptText);
  const reason = `Agent call blocked: missing subagent_type (defaults to general-purpose with all tools).

Always specify a specialist agent type. Based on your prompt, consider:
${suggestions.join("\n")}

Full agent list: scout, oracle, kraken, spark, sleuth, aegis, architect, phoenix, critic, scribe, arbiter, atlas, profiler, herald, maestro

Re-run with subagent_type set to the appropriate specialist.`;
  const modelContext = `Blocked: general-purpose agent. Use a specialist: ${suggestions.map((s) => s.replace(/^- /, "")).join(", ")}`;
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: reason,
      additionalContext: modelContext
    }
  };
}
function processInput(input) {
  if (input.tool_name !== "Agent") {
    return {};
  }
  const subagentType = input.tool_input?.subagent_type;
  if (isSubagentTypeValid(subagentType)) {
    return {};
  }
  const promptText = input.tool_input?.prompt ?? input.tool_input?.description ?? "";
  return buildDenyResponse(promptText);
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
  const result = processInput(input);
  console.log(JSON.stringify(result));
}
main().catch(console.error);
export {
  AGENT_SUGGESTIONS,
  buildDenyResponse,
  isSubagentTypeValid,
  processInput,
  suggestAgents
};
