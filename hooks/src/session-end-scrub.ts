//! @hook SessionEnd @preserve
// Detach-spawns the transcript scrubber so it survives the 60s SessionEnd timeout.
// Always emits `{result:'continue'}` and exits within ~50ms.

import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";

interface SessionEndInput {
  session_id: string;
  transcript_path: string;
  reason: "clear" | "logout" | "prompt_input_exit" | "other";
}

async function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.on("data", (c) => (data += c));
    process.stdin.on("end", () => resolve(data));
  });
}

async function main() {
  try {
    const raw = await readStdin();
    const input: SessionEndInput = JSON.parse(raw);

    if (input.transcript_path && fs.existsSync(input.transcript_path)) {
      const scrubber = path.join(
        process.env.HOME || "",
        ".claude",
        "hooks",
        "dist",
        "scrub-transcript.mjs",
      );

      if (fs.existsSync(scrubber)) {
        const child = spawn("node", [scrubber, input.transcript_path], {
          detached: true,
          stdio: "ignore",
          env: { ...process.env, SCRUB_TRIGGER: "sessionend" },
        });
        child.unref();
      }
    }
  } catch {
    // Never block session end on errors.
  }
  console.log(JSON.stringify({ result: "continue" }));
}

main();
