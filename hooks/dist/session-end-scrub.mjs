// src/session-end-scrub.ts
import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";
//! @hook SessionEnd @preserve
async function readStdin() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.on("data", (c) => data += c);
    process.stdin.on("end", () => resolve(data));
  });
}
async function main() {
  try {
    const raw = await readStdin();
    const input = JSON.parse(raw);
    if (input.transcript_path && fs.existsSync(input.transcript_path)) {
      const scrubber = path.join(
        process.env.HOME || "",
        ".claude",
        "hooks",
        "dist",
        "scrub-transcript.mjs"
      );
      if (fs.existsSync(scrubber)) {
        const child = spawn("node", [scrubber, input.transcript_path], {
          detached: true,
          stdio: "ignore",
          env: { ...process.env, SCRUB_TRIGGER: "sessionend" }
        });
        child.unref();
      }
    }
  } catch {
  }
  console.log(JSON.stringify({ result: "continue" }));
}
main();
