#!/usr/bin/env node

// src/task-router.ts
import { readFileSync } from "fs";
import { spawnSync } from "child_process";
function makeHookOutput(decision, reason) {
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: decision,
      permissionDecisionReason: reason
    }
  };
}
function allowTask() {
  return makeHookOutput("allow");
}
function denyTask(reason) {
  return makeHookOutput("deny", reason);
}
function extractPrompt(toolInput) {
  if (typeof toolInput.prompt === "string") {
    return toolInput.prompt;
  }
  if (typeof toolInput.description === "string") {
    return toolInput.description;
  }
  return null;
}
function callUnifiedGate(prompt, sessionId) {
  const escapedPrompt = prompt.replace(/'/g, "'\\''");
  const escapedSessionId = sessionId.replace(/'/g, "'\\''");
  const pythonCode = `
import asyncio
import json
import sys

try:
    from scripts.agentica_patterns.unified_gate import (
        orchestrate_with_gates,
        ClarificationRequired,
        ResourcesInsufficient,
        CompositionInvalid,
    )

    async def main():
        try:
            result = await orchestrate_with_gates(
                prompt='''${escapedPrompt}''',
                session_id='${escapedSessionId}',
            )
            print(json.dumps({
                'success': result.get('success', False),
                'pattern_used': result.get('pattern_used'),
                'confidence': result.get('confidence', 0.5),
                'session_id': result.get('session_id'),
                'aggregated_output': result.get('aggregated_output'),
                'errors': result.get('errors', []),
                'total_duration_ms': result.get('total_duration_ms'),
            }))
        except ClarificationRequired as e:
            print(json.dumps({
                'success': False,
                'gate_failure': {
                    'type': 'ClarificationRequired',
                    'unknowns': e.unknowns,
                },
            }))
        except ResourcesInsufficient as e:
            print(json.dumps({
                'success': False,
                'gate_failure': {
                    'type': 'ResourcesInsufficient',
                    'bottleneck': e.bottleneck,
                    'suggestion': e.suggestion,
                },
            }))
        except CompositionInvalid as e:
            print(json.dumps({
                'success': False,
                'gate_failure': {
                    'type': 'CompositionInvalid',
                    'errors': e.errors,
                },
            }))
        except Exception as e:
            print(json.dumps({
                'success': False,
                'errors': [str(e)],
            }))

    asyncio.run(main())
except ImportError as e:
    # Module not available - return success to allow vanilla Task
    import json
    print(json.dumps({'success': True, 'pattern_used': None}))
except Exception as e:
    import json
    print(json.dumps({'success': False, 'errors': [str(e)]}))
`;
  try {
    const result = spawnSync("uv", ["run", "python", "-c", pythonCode], {
      cwd: process.env.CLAUDE_PROJECT_DIR || process.cwd(),
      encoding: "utf-8",
      timeout: 3e4,
      // 30 second timeout
      env: {
        ...process.env,
        PYTHONPATH: process.env.CLAUDE_PROJECT_DIR || process.cwd()
      }
    });
    if (result.error) {
      console.error("[task-router] Failed to spawn Python:", result.error);
      return null;
    }
    if (result.status !== 0) {
      console.error("[task-router] Python exited with status:", result.status);
      console.error("[task-router] stderr:", result.stderr);
      return null;
    }
    const stdout = result.stdout.trim();
    if (!stdout) {
      return null;
    }
    return JSON.parse(stdout);
  } catch (err) {
    console.error("[task-router] Error calling unified gate:", err);
    return null;
  }
}
function callAgentRouter(prompt) {
  const escapedPrompt = prompt.replace(/'/g, "'\\''");
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const opcDir = process.env.CLAUDE_OPC_DIR || `${projectDir}/opc`;
  try {
    const result = spawnSync("uv", ["run", "python", "-m", "scripts.agentica_patterns.agent_router", escapedPrompt], {
      cwd: opcDir,
      encoding: "utf-8",
      timeout: 5e3,
      // 5 second timeout - keyword matching is instant
      env: {
        ...process.env,
        PYTHONPATH: opcDir
      }
    });
    if (result.error || result.status !== 0) {
      return null;
    }
    const stdout = result.stdout.trim();
    if (!stdout) {
      return null;
    }
    return JSON.parse(stdout);
  } catch {
    return null;
  }
}
async function main() {
  let input;
  try {
    const rawInput = readFileSync(0, "utf-8");
    if (!rawInput.trim()) {
      console.log(JSON.stringify(allowTask()));
      return;
    }
    input = JSON.parse(rawInput);
  } catch (err) {
    console.log(JSON.stringify(allowTask()));
    return;
  }
  if (input.tool_name !== "Task") {
    console.log(JSON.stringify(allowTask()));
    return;
  }
  const prompt = extractPrompt(input.tool_input);
  if (!prompt) {
    console.log(JSON.stringify(allowTask()));
    return;
  }
  const sessionId = input.session_id || process.env.CLAUDE_SESSION_ID || "default";
  const gateResult = callUnifiedGate(prompt, sessionId);
  let unknowns = [];
  if (gateResult?.gate_failure) {
    const failure = gateResult.gate_failure;
    switch (failure.type) {
      case "ClarificationRequired":
        unknowns = failure.unknowns || [];
        break;
      case "ResourcesInsufficient":
        console.log(JSON.stringify(denyTask(
          `Insufficient resources: ${failure.bottleneck} is the bottleneck. Suggestion: ${failure.suggestion || "Reduce agent count or simplify the task."}`
        )));
        return;
      case "CompositionInvalid":
        console.log(JSON.stringify(denyTask(
          `Invalid pattern composition: ${failure.errors?.join("; ") || "composition rules violated"}. Please restructure the task.`
        )));
        return;
    }
  }
  const routingResult = callAgentRouter(prompt);
  const agent = routingResult?.agent ?? "kraken";
  const confidence = routingResult?.confidence ?? 0.3;
  const method = routingResult?.method ?? "fallback";
  let finalPrompt = prompt;
  if (unknowns.length > 0) {
    finalPrompt = `${prompt}

[CLARIFY FIRST using AskUserQuestion: ${unknowns.join(", ")}]`;
  }
  const escapedPrompt = finalPrompt.replace(/'/g, "'\\''").replace(/\n/g, " ");
  let routingMsg = `\u2192 Routing to **${agent}** (${(confidence * 100).toFixed(0)}% confidence, ${method})`;
  if (unknowns.length > 0) {
    routingMsg += `
\u26A0\uFE0F Agent will clarify: ${unknowns.join(", ")}`;
  }
  const reason = `${routingMsg}

**Run:**
\`\`\`bash
uv run python -c "from scripts.claude_spawn import spawn_agent, route_to_agent; spawn_agent('${escapedPrompt}', perspective=route_to_agent('${escapedPrompt}'))"
\`\`\``;
  console.log(JSON.stringify(denyTask(reason)));
}
main().catch((err) => {
  console.error("[task-router] Uncaught error:", err);
  console.log(JSON.stringify(allowTask()));
});
