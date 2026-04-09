#!/usr/bin/env node

// src/skill-activation-prompt.ts
import { readFileSync as readFileSync2, existsSync as existsSync2 } from "fs";
import { join as join2 } from "path";
import { spawnSync } from "child_process";
import { tmpdir as tmpdir2 } from "os";

// src/shared/resource-reader.ts
import { readFileSync, existsSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
var DEFAULT_RESOURCE_STATE = {
  freeMemMB: 4096,
  activeAgents: 0,
  maxAgents: 10,
  contextPct: 0
};
function getSessionId() {
  return process.env.CLAUDE_SESSION_ID || String(process.ppid || process.pid);
}
function getResourceFilePath(sessionId) {
  return join(tmpdir(), `claude-resources-${sessionId}.json`);
}
function readResourceState() {
  const sessionId = getSessionId();
  const resourceFile = getResourceFilePath(sessionId);
  if (!existsSync(resourceFile)) {
    return null;
  }
  try {
    const content = readFileSync(resourceFile, "utf-8");
    const data = JSON.parse(content);
    return {
      freeMemMB: typeof data.freeMemMB === "number" ? data.freeMemMB : DEFAULT_RESOURCE_STATE.freeMemMB,
      activeAgents: typeof data.activeAgents === "number" ? data.activeAgents : DEFAULT_RESOURCE_STATE.activeAgents,
      maxAgents: typeof data.maxAgents === "number" ? data.maxAgents : DEFAULT_RESOURCE_STATE.maxAgents,
      contextPct: typeof data.contextPct === "number" ? data.contextPct : DEFAULT_RESOURCE_STATE.contextPct
    };
  } catch {
    return null;
  }
}

// src/skill-validation-prompt.ts
/*!
 * Prompt-Based Skill Validation
 *
 * Reduces false-positive skill activations by using LLM validation to
 * distinguish between:
 * - "mentions keyword" (e.g., "commit" in "I need to commit to this approach")
 * - "actually needs this skill" (e.g., "commit these changes to git")
 *
 * This module provides:
 * 1. Heuristics to determine when LLM validation is needed
 * 2. Prompt templates for validation
 * 3. Response parsing utilities
 */
var AMBIGUOUS_KEYWORDS = /* @__PURE__ */ new Set([
  "commit",
  "push",
  "pull",
  "merge",
  "branch",
  "checkout",
  "debug",
  "build",
  "implement",
  "plan",
  "research",
  "deploy",
  "release",
  "fix",
  "test",
  "validate",
  "review",
  "analyze",
  "document",
  "refactor",
  "optimize"
]);
var SPECIFIC_TECHNICAL_TERMS = /* @__PURE__ */ new Set([
  "sympy",
  "braintrust",
  "perplexity",
  "agentica",
  "firecrawl",
  "qlty",
  "repoprompt",
  "ast-grep",
  "morph",
  "ragie",
  "lean4",
  "mathlib",
  "z3",
  "shapely",
  "pint"
]);
var TECHNICAL_CONTEXT_INDICATORS = {
  commit: ["git", "changes", "files", "message", "push", "repository", "branch", "staged"],
  push: ["git", "remote", "origin", "branch", "repository", "upstream"],
  pull: ["git", "remote", "origin", "branch", "merge", "rebase", "request"],
  merge: ["git", "branch", "conflict", "pull request", "pr"],
  branch: ["git", "checkout", "create", "switch", "feature"],
  checkout: ["git", "branch", "file", "commit", "HEAD"],
  debug: ["error", "bug", "issue", "logs", "stack trace", "exception", "crash", "breakpoint"],
  build: ["npm", "yarn", "cargo", "make", "compile", "webpack", "bundle", "project"],
  implement: ["code", "feature", "function", "class", "method", "api", "interface", "module"],
  plan: ["implementation", "phase", "architecture", "design", "roadmap", "milestone"],
  research: ["api", "library", "documentation", "docs", "best practices", "pattern", "codebase"],
  deploy: ["server", "production", "staging", "kubernetes", "docker", "cloud", "ci/cd"],
  release: ["version", "tag", "changelog", "npm", "package", "publish"],
  fix: ["bug", "error", "issue", "broken", "failing", "test", "regression"],
  test: ["unit", "integration", "e2e", "coverage", "spec", "jest", "pytest", "vitest"],
  validate: ["input", "schema", "data", "form", "field", "type"],
  review: ["code", "pr", "pull request", "changes", "diff"],
  analyze: ["code", "codebase", "performance", "metrics", "logs"],
  document: ["api", "readme", "docs", "jsdoc", "docstring", "comments"],
  refactor: ["code", "function", "class", "module", "clean up", "simplify"],
  optimize: ["performance", "speed", "memory", "query", "algorithm"]
};
function shouldValidateWithLLM(match) {
  if (match.matchType === "explicit") {
    return false;
  }
  if (match.enforcement === "block") {
    return false;
  }
  if (match.matchType === "intent") {
    return false;
  }
  const termLower = match.matchedTerm.toLowerCase();
  if (SPECIFIC_TECHNICAL_TERMS.has(termLower)) {
    return false;
  }
  if (match.matchType === "keyword" && AMBIGUOUS_KEYWORDS.has(termLower)) {
    const promptLower = match.prompt.toLowerCase();
    const technicalIndicators = TECHNICAL_CONTEXT_INDICATORS[termLower] || [];
    for (const indicator of technicalIndicators) {
      const regex = new RegExp(`\\b${indicator.toLowerCase()}\\b`);
      if (regex.test(promptLower)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// src/skill-activation-prompt.ts
var PATTERN_AGENT_MAP = {
  "swarm": "research-agent",
  "hierarchical": "kraken",
  "pipeline": "kraken",
  "generator_critic": "review-agent",
  "adversarial": "validate-agent",
  "map_reduce": "kraken",
  "jury": "validate-agent",
  "blackboard": "maestro",
  "circuit_breaker": "kraken",
  "chain_of_responsibility": "maestro",
  "event_driven": "kraken"
};
function runPatternInference(prompt, projectDir) {
  try {
    const scriptPath = join2(projectDir, "scripts", "agentica_patterns", "pattern_inference.py");
    if (!existsSync2(scriptPath)) {
      return null;
    }
    const pythonCode = `
import sys
import json
import importlib.util

# Direct import bypassing __init__.py
spec = importlib.util.spec_from_file_location(
    'pattern_inference',
    ${JSON.stringify(scriptPath)}
)
pattern_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pattern_mod)

prompt = ${JSON.stringify(prompt)}
result = pattern_mod.infer_pattern(prompt)
output = result.to_dict()
output['work_breakdown_detailed'] = pattern_mod.generate_work_breakdown(result)
print(json.dumps(output))
`;
    const result = spawnSync("uv", ["run", "python", "-c", pythonCode], {
      encoding: "utf-8",
      timeout: 5e3,
      cwd: projectDir,
      stdio: ["pipe", "pipe", "pipe"]
    });
    if (result.status !== 0 || !result.stdout) {
      return null;
    }
    return JSON.parse(result.stdout.trim());
  } catch (err) {
    return null;
  }
}
async function main() {
  try {
    const input = readFileSync2(0, "utf-8");
    let data;
    try {
      data = JSON.parse(input);
    } catch {
      process.exit(0);
    }
    if (!data.prompt || typeof data.prompt !== "string") {
      process.exit(0);
    }
    const prompt = data.prompt.toLowerCase();
    const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
    const homeDir = process.env.HOME || process.env.USERPROFILE || "";
    const projectRulesPath = join2(projectDir, ".claude", "skills", "skill-rules.json");
    const globalRulesPath = join2(homeDir, ".claude", "skills", "skill-rules.json");
    let rulesPath = "";
    if (existsSync2(projectRulesPath)) {
      rulesPath = projectRulesPath;
    } else if (existsSync2(globalRulesPath)) {
      rulesPath = globalRulesPath;
    } else {
      process.exit(0);
    }
    const rules = JSON.parse(readFileSync2(rulesPath, "utf-8"));
    const wordCount = data.prompt.split(/\s+/).length;
    const patternInference = wordCount >= 20 ? runPatternInference(data.prompt, projectDir) : null;
    const matchedSkills = [];
    for (const [skillName, config] of Object.entries(rules.skills)) {
      const triggers = config.promptTriggers;
      if (!triggers) {
        continue;
      }
      if (triggers.keywords) {
        const matchedKeyword = triggers.keywords.find(
          (kw) => prompt.includes(kw.toLowerCase())
        );
        if (matchedKeyword) {
          const skillMatchForValidation = {
            skillName,
            matchType: "keyword",
            matchedTerm: matchedKeyword,
            prompt: data.prompt,
            // Use original prompt (not lowercased)
            skillDescription: config.description,
            enforcement: config.enforcement
          };
          const needsValidation = shouldValidateWithLLM(skillMatchForValidation);
          matchedSkills.push({
            name: skillName,
            matchType: "keyword",
            matchedTerm: matchedKeyword,
            config,
            needsValidation
          });
          continue;
        }
      }
      if (triggers.intentPatterns) {
        const intentMatch = triggers.intentPatterns.some((pattern) => {
          try {
            const regex = new RegExp(pattern, "i");
            return regex.test(prompt);
          } catch {
            return false;
          }
        });
        if (intentMatch) {
          matchedSkills.push({
            name: skillName,
            matchType: "intent",
            config,
            needsValidation: false
          });
        }
      }
    }
    const matchedAgents = [];
    if (rules.agents) {
      for (const [agentName, config] of Object.entries(rules.agents)) {
        const triggers = config.promptTriggers;
        if (!triggers) {
          continue;
        }
        if (triggers.keywords) {
          const matchedKeyword = triggers.keywords.find(
            (kw) => prompt.includes(kw.toLowerCase())
          );
          if (matchedKeyword) {
            const skillMatchForValidation = {
              skillName: agentName,
              matchType: "keyword",
              matchedTerm: matchedKeyword,
              prompt: data.prompt,
              skillDescription: config.description,
              enforcement: config.enforcement
            };
            const needsValidation = shouldValidateWithLLM(skillMatchForValidation);
            matchedAgents.push({
              name: agentName,
              matchType: "keyword",
              matchedTerm: matchedKeyword,
              config,
              isAgent: true,
              needsValidation
            });
            continue;
          }
        }
        if (triggers.intentPatterns) {
          const intentMatch = triggers.intentPatterns.some((pattern) => {
            try {
              const regex = new RegExp(pattern, "i");
              return regex.test(prompt);
            } catch {
              return false;
            }
          });
          if (intentMatch) {
            matchedAgents.push({
              name: agentName,
              matchType: "intent",
              config,
              isAgent: true,
              needsValidation: false
            });
          }
        }
      }
    }
    const confirmedSkills = matchedSkills.filter((s) => !s.needsValidation);
    const confirmedAgents = matchedAgents.filter((a) => !a.needsValidation);
    const showPatternInference = patternInference && patternInference.confidence >= 0.7 && data.prompt.split(/\s+/).length >= 20;
    const blockingSkills = confirmedSkills.filter((s) => s.config.enforcement === "block");
    if (confirmedSkills.length === 0 && confirmedAgents.length === 0 && !showPatternInference && blockingSkills.length === 0) {
    } else {
      let output = "";
      if (showPatternInference && patternInference) {
        const suggestedAgent = PATTERN_AGENT_MAP[patternInference.pattern] || "kraken";
        output += `PATTERN: ${patternInference.pattern} \u2192 ${suggestedAgent} (${Math.round(patternInference.confidence * 100)}%)
`;
      }
      if (confirmedSkills.length > 0 || confirmedAgents.length > 0) {
        const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
        const allConfirmed = [
          ...confirmedSkills.map((s) => ({ ...s, sortKey: priorityOrder[s.config.priority] ?? 3, isAgent: false })),
          ...confirmedAgents.map((a) => ({ ...a, sortKey: priorityOrder[a.config.priority] ?? 3, isAgent: true }))
        ];
        allConfirmed.sort((a, b) => {
          if (a.sortKey !== b.sortKey) return a.sortKey - b.sortKey;
          if (a.matchType === "intent" && b.matchType !== "intent") return -1;
          if (b.matchType === "intent" && a.matchType !== "intent") return 1;
          return 0;
        });
        const MAX_SUGGESTIONS = 5;
        const capped = allConfirmed.slice(0, MAX_SUGGESTIONS);
        const skills = capped.filter((s) => !s.isAgent);
        const agents = capped.filter((s) => s.isAgent);
        if (skills.length > 0) {
          const hasBlock = skills.some((s) => s.config.enforcement === "block");
          output += hasBlock ? "REQUIRED: " : "Skills: ";
          output += skills.map((s) => s.name).join(", ") + "\n";
        }
        if (agents.length > 0) {
          output += "Agents: " + agents.map((a) => a.name).join(", ") + "\n";
        }
      }
      if (blockingSkills.length > 0) {
        const blockMessage = `BLOCKING: Invoke ${blockingSkills.map((s) => s.name).join(", ")} before responding.
` + output;
        console.log(JSON.stringify({
          result: "block",
          reason: blockMessage
        }));
        process.exit(0);
      }
      if (output) {
        console.log(output.trimEnd());
      }
    }
    const rawSessionId = data.session_id || process.env.CLAUDE_SESSION_ID || process.env.CLAUDE_PPID || "default";
    const sessionId = rawSessionId.slice(0, 8);
    const contextFile = join2(tmpdir2(), `claude-context-pct-${sessionId}.txt`);
    if (existsSync2(contextFile)) {
      try {
        const pct = parseInt(readFileSync2(contextFile, "utf-8").trim(), 10);
        let contextWarning = "";
        if (pct >= 90) {
          contextWarning = "\n" + "=".repeat(50) + "\n  CONTEXT CRITICAL: " + pct + "%\n  Run /create_handoff NOW before auto-compact!\n" + "=".repeat(50) + "\n";
        } else if (pct >= 80) {
          contextWarning = "\nCONTEXT WARNING: " + pct + "%\nRecommend: /create_handoff then /clear soon\n";
        } else if (pct >= 70) {
          contextWarning = "\nContext at " + pct + "%. Consider handoff when you reach a stopping point.\n";
        }
        if (contextWarning) {
          console.log(contextWarning);
        }
      } catch {
      }
    }
    const resources = readResourceState();
    if (resources && resources.maxAgents > 0) {
      const utilization = resources.activeAgents / resources.maxAgents;
      let resourceWarning = "";
      if (utilization >= 1) {
        resourceWarning = "\n" + "=".repeat(50) + "\nRESOURCE CRITICAL: At limit (" + resources.activeAgents + "/" + resources.maxAgents + " agents)\nDo NOT spawn new agents until existing ones complete.\n" + "=".repeat(50) + "\n";
      } else if (utilization >= 0.8) {
        const remaining = resources.maxAgents - resources.activeAgents;
        resourceWarning = "\nRESOURCE WARNING: Near limit (" + resources.activeAgents + "/" + resources.maxAgents + " agents)\nOnly " + remaining + " agent slot(s) remaining. Limit spawning.\n";
      }
      if (resourceWarning) {
        console.log(resourceWarning);
      }
    }
    process.exit(0);
  } catch (err) {
    console.error("Error in skill-activation-prompt hook:", err);
    process.exit(1);
  }
}
main().catch((err) => {
  console.error("Uncaught error:", err);
  process.exit(1);
});
