// src/stop-learnings.ts
import * as fs from "fs";
function readStdin() {
  try {
    return fs.readFileSync(0, "utf-8");
  } catch {
    return "{}";
  }
}
function countSignificantWork(transcriptPath) {
  let edits = 0;
  let turns = 0;
  let tools = 0;
  try {
    if (!fs.existsSync(transcriptPath)) {
      return { edits: 0, turns: 0, tools: 0 };
    }
    const content = fs.readFileSync(transcriptPath, "utf-8");
    const lines = content.trim().split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.type === "user") {
          const content2 = msg.message?.content;
          if (typeof content2 === "string" || Array.isArray(content2) && content2[0]?.type !== "tool_result") {
            turns++;
          }
        }
        if (msg.type === "assistant") {
          const content2 = msg.message?.content;
          if (Array.isArray(content2)) {
            for (const block of content2) {
              if (block.type === "tool_use") {
                tools++;
                const toolName = block.name || "";
                if (["Edit", "Write", "MultiEdit", "NotebookEdit"].includes(toolName)) {
                  edits++;
                }
              }
            }
          }
        }
      } catch {
      }
    }
  } catch {
  }
  return { edits, turns, tools };
}
async function main() {
  const input = JSON.parse(readStdin());
  if (input.stop_hook_active) {
    console.log(JSON.stringify({}));
    return;
  }
  const work = countSignificantWork(input.transcript_path);
  const hasSignificantWork = work.edits >= 2 || work.turns >= 3 || work.tools >= 5;
  if (!hasSignificantWork) {
    console.log(JSON.stringify({}));
    return;
  }
  const sessionId = input.session_id;
  const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd;
  const opcDir = process.env.CLAUDE_OPC_DIR || projectDir;
  const prompt = `Before ending this session, extract key learnings for future reference.

**Instructions:**
1. Reflect on this session's work
2. Identify 3-5 key learnings in these categories:
   - **worked**: What approaches/techniques succeeded
   - **failed**: What didn't work or was tricky
   - **decisions**: Key choices made and rationale
   - **patterns**: Reusable techniques for similar tasks

3. Store the learnings by running this command:

```bash
cd $CLAUDE_OPC_DIR && uv run python scripts/core/store_learning.py \\
  --session-id "${sessionId}" \\
  --worked "..." \\
  --failed "..." \\
  --decisions "..." \\
  --patterns "..."
\`\`\`

Replace "..." with actual content. Keep each field concise (1-3 sentences).
If a category doesn't apply, use "None" for that field.

This stores learnings in the memory system for future recall.`;
  const output = {
    decision: "block",
    reason: prompt
  };
  console.log(JSON.stringify(output));
}
main().catch((err) => {
  console.error("stop-learnings error:", err);
  console.log(JSON.stringify({}));
});
