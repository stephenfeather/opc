/**
 * Stop Learnings Hook - Claude-native session learning extraction
 *
 * Fires at session Stop to prompt Claude to extract learnings from its own context.
 * No external dependencies (Braintrust, etc.) - works anywhere.
 *
 * Flow:
 * 1. Stop hook fires
 * 2. Check stop_hook_active to prevent infinite loops
 * 3. Read transcript to assess if significant work was done
 * 4. If yes, block and prompt Claude to extract learnings
 * 5. Claude outputs learnings and stores via Bash command
 */
import * as fs from 'fs';
function readStdin() {
    try {
        return fs.readFileSync(0, 'utf-8');
    }
    catch {
        return '{}';
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
        const content = fs.readFileSync(transcriptPath, 'utf-8');
        const lines = content.trim().split('\n');
        for (const line of lines) {
            if (!line.trim())
                continue;
            try {
                const msg = JSON.parse(line);
                if (msg.type === 'user') {
                    // Count real user turns (not tool results)
                    const content = msg.message?.content;
                    if (typeof content === 'string' ||
                        (Array.isArray(content) && content[0]?.type !== 'tool_result')) {
                        turns++;
                    }
                }
                if (msg.type === 'assistant') {
                    const content = msg.message?.content;
                    if (Array.isArray(content)) {
                        for (const block of content) {
                            if (block.type === 'tool_use') {
                                tools++;
                                const toolName = block.name || '';
                                if (['Edit', 'Write', 'MultiEdit', 'NotebookEdit'].includes(toolName)) {
                                    edits++;
                                }
                            }
                        }
                    }
                }
            }
            catch {
                // Skip malformed lines
            }
        }
    }
    catch {
        // If transcript unreadable, assume no significant work
    }
    return { edits, turns, tools };
}
async function main() {
    const input = JSON.parse(readStdin());
    // CRITICAL: Prevent infinite loops
    if (input.stop_hook_active) {
        console.log(JSON.stringify({}));
        return;
    }
    // Check for significant work
    const work = countSignificantWork(input.transcript_path);
    // Thresholds for "significant work"
    const hasSignificantWork = work.edits >= 2 || work.turns >= 3 || work.tools >= 5;
    if (!hasSignificantWork) {
        // Not enough work to warrant learning extraction
        console.log(JSON.stringify({}));
        return;
    }
    // Build the learning extraction prompt
    const sessionId = input.session_id;
    const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd;
	const opcDir = process.env.CLAUDE_OPC_DIR || '';
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
cd $CLAUDE_OPC_DIR && uv run python scripts/store_learning.py \\
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
        decision: 'block',
        reason: prompt
    };
    console.log(JSON.stringify(output));
}
main().catch(err => {
    console.error('stop-learnings error:', err);
    console.log(JSON.stringify({}));
});
