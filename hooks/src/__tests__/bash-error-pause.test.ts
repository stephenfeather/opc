import { describe, expect, it } from "vitest";
import { spawnSync } from "child_process";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BIN = path.resolve(__dirname, "..", "..", "dist", "bash-error-pause.mjs");

function run(input: string) {
  return spawnSync("node", [BIN], {
    encoding: "utf-8",
    input,
  });
}

describe("bash-error-pause CLI input handling", () => {
  it("continues for non-object JSON", () => {
    const res = run("null");
    expect(res.status).toBe(0);
    expect(JSON.parse(res.stdout.trim()).result).toBe("continue");
  });

  it("preserves Bash warning detection for valid payloads", () => {
    const res = run(
      JSON.stringify({
        tool_name: "Bash",
        tool_response: { stderr: "warning: deprecated option\n" },
      }),
    );

    expect(res.status).toBe(0);
    const output = JSON.parse(res.stdout.trim());
    expect(output.result).toBe("continue");
    expect(output.hookSpecificOutput.additionalContext).toContain("WARNING detected");
  });
});
