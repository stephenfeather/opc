import { describe, expect, it } from "vitest";
import { spawnSync } from "child_process";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HOOK_BIN = path.resolve(__dirname, "..", "..", "dist", "read-secret-block.mjs");

function decision(filePath: string): string {
  const res = spawnSync("node", [HOOK_BIN], {
    encoding: "utf-8",
    input: JSON.stringify({ tool_name: "Read", tool_input: { file_path: filePath } }),
  });
  if (!res.stdout.trim()) return "allow";
  try {
    return JSON.parse(res.stdout).hookSpecificOutput?.permissionDecision ?? "allow";
  } catch {
    return "malformed";
  }
}

describe("read-secret-block path matching", () => {
  it("denies POSIX credential paths", () => {
    expect(decision("/Users/x/.config/gh/hosts.yml")).toBe("deny");
    expect(decision("/Users/x/.aws/credentials")).toBe("deny");
  });

  it("denies Windows backslash credential paths (separator bypass)", () => {
    expect(decision("C:\\Users\\x\\.config\\gh\\hosts.yml")).toBe("deny");
    expect(decision("C:\\Users\\x\\.config\\doctl\\config.yaml")).toBe("deny");
    expect(decision("C:\\Users\\x\\.aws\\credentials")).toBe("deny");
  });

  it("still allows harmless files either way", () => {
    expect(decision("/Users/x/p/tsconfig.json")).toBe("allow");
    expect(decision("C:\\Users\\x\\p\\tsconfig.json")).toBe("allow");
  });
});
