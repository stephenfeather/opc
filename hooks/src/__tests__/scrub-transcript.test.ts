import { describe, expect, it } from "vitest";
import { spawnSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRUB_BIN = path.resolve(__dirname, "..", "..", "dist", "scrub-transcript.mjs");

function runScrub(transcriptPath: string) {
  return spawnSync("node", [SCRUB_BIN, transcriptPath], { encoding: "utf-8" });
}

function makeTranscript(lines: string[]): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "scrub-test-"));
  const p = path.join(dir, "transcript.jsonl");
  fs.writeFileSync(p, lines.join("\n") + "\n");
  return p;
}

// Token assembled at runtime so the test source never contains a
// contiguous secret-shaped literal (GitHub push protection).
const FAKE_DO_TOKEN = "dop_v1_" + "0123456789abcdef".repeat(4);

describe("scrub-transcript CLI", () => {
  it("redacts a secret-bearing transcript", () => {
    const p = makeTranscript([
      JSON.stringify({ role: "user", content: `token is ${FAKE_DO_TOKEN}` }),
    ]);
    const res = runScrub(p);
    expect(res.status).toBe(0);
    const after = fs.readFileSync(p, "utf-8");
    expect(after).not.toContain(FAKE_DO_TOKEN);
    expect(after).toContain("[REDACTED:");
  });

  it("does not rewrite or touch mtime when nothing is redacted", () => {
    const p = makeTranscript([
      JSON.stringify({ role: "user", content: "hello world, nothing sensitive" }),
    ]);
    const past = new Date(Date.now() - 60_000);
    fs.utimesSync(p, past, past);
    const mtimeBefore = fs.statSync(p).mtimeMs;

    const res = runScrub(p);
    expect(res.status).toBe(0);

    const mtimeAfter = fs.statSync(p).mtimeMs;
    expect(mtimeAfter).toBe(mtimeBefore);
    expect(fs.existsSync(`${p}.scrub.tmp`)).toBe(false);
  });

  it("preserves a restrictive file mode across the scrub rewrite", () => {
    const p = makeTranscript([
      JSON.stringify({ role: "user", content: `token is ${FAKE_DO_TOKEN}` }),
    ]);
    fs.chmodSync(p, 0o600);
    const res = runScrub(p);
    expect(res.status).toBe(0);
    expect(fs.readFileSync(p, "utf-8")).toContain("[REDACTED:");
    expect(fs.statSync(p).mode & 0o777).toBe(0o600);
  });

  it("is idempotent: scrubbing twice leaves the file stable", () => {
    const p = makeTranscript([
      JSON.stringify({ role: "user", content: `token is ${FAKE_DO_TOKEN}` }),
    ]);
    expect(runScrub(p).status).toBe(0);
    const once = fs.readFileSync(p, "utf-8");
    const stamp = fs.statSync(p).mtimeMs;
    expect(runScrub(p).status).toBe(0);
    expect(fs.readFileSync(p, "utf-8")).toBe(once);
    // Second run found nothing to redact, so the file must be untouched.
    expect(fs.statSync(p).mtimeMs).toBe(stamp);
  });
});
