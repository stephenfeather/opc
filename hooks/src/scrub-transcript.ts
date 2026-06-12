// Scrubs a single Claude Code JSONL transcript: replaces credential matches
// with `[REDACTED:<rule-id>]`, atomic-renames over the original.
//
// Usage: node dist/scrub-transcript.mjs <transcript-path>
//
// Safe properties:
//   - Line-by-line streaming: handles arbitrarily large transcripts.
//   - Substring replacement on the raw line: preserves JSON validity because
//     secret values never contain unescaped JSON delimiters (rules anchor on
//     prefixes / charsets that exclude `"` `\` `\n`).
//   - Atomic: writes to `<path>.scrub.tmp`, then rename() over original.
//   - Idempotent: scanning a redacted transcript finds nothing; second run is a no-op.

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { scanAll } from "./credential-scanner/scan-all.ts";

const AUDIT_LOG = path.join(
  process.env.HOME || "",
  ".claude",
  "hooks",
  "scrub-audit.log",
);

function appendAudit(transcript: string, perRule: Map<string, number>): void {
  if (perRule.size === 0) return;
  const ts = new Date().toISOString();
  const trigger = process.env.SCRUB_TRIGGER || "manual";
  const lines: string[] = [];
  for (const [ruleId, count] of perRule) {
    lines.push(
      JSON.stringify({ ts, trigger, transcript, rule_id: ruleId, count }) + "\n",
    );
  }
  try {
    fs.appendFileSync(AUDIT_LOG, lines.join(""));
  } catch {
    // Audit log failures must never break the scrub.
  }
}

async function scrub(transcriptPath: string): Promise<void> {
  if (!fs.existsSync(transcriptPath)) {
    process.stderr.write(`scrub-transcript: not found: ${transcriptPath}\n`);
    process.exit(1);
  }

  // Per-file lock (sibling of transcript) so the daily cron and SessionEnd
  // hook can't race on the same JSONL.
  const lockPath = `${transcriptPath}.scrub.lock`;
  let lockFd: number;
  try {
    lockFd = fs.openSync(lockPath, "wx");
  } catch {
    // Another scrubber owns this file; skip silently.
    return;
  }
  fs.writeSync(lockFd, `${process.pid}:${Date.now()}\n`);

  const tmpPath = `${transcriptPath}.scrub.tmp`;
  // Clean stale tmp from a prior crashed run.
  try { fs.unlinkSync(tmpPath); } catch { /* ignore */ }

  // Preserve the transcript's mode: a default-mode temp file renamed over a
  // 0600 transcript would widen access to freshly-redacted session history.
  const srcMode = fs.statSync(transcriptPath).mode & 0o7777;
  const out = fs.createWriteStream(tmpPath, { flags: "wx", mode: srcMode });
  const reader = readline.createInterface({
    input: fs.createReadStream(transcriptPath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  let redactionCount = 0;
  const perRule = new Map<string, number>();

  try {
    for await (const line of reader) {
      let scrubbed = line;
      const findings = scanAll(line);
      for (const f of findings) {
        // Skip if the match is already a redaction marker — keeps the rule-id
        // stable across re-runs and makes the scrubber strictly idempotent.
        if (f.secretValue.startsWith("[REDACTED:")) continue;
        // Most rule charsets exclude `"` and `\`, but the vendor
        // `env-var-assignment` rule captures `\S{8,}` greedily and can
        // swallow a JSON closing quote. Trim at the first JSON-boundary
        // character so the replacement never destroys structural quotes.
        let secret = f.secretValue;
        if (f.ruleId === "private-key") {
          // PEM blocks legitimately span \n escapes inside the JSON string;
          // trim only at a structural quote so the key material is removed.
          const q = secret.indexOf('"');
          if (q >= 0) secret = secret.slice(0, q);
        } else {
          const boundary = secret.search(/["\\]/);
          if (boundary >= 0) secret = secret.slice(0, boundary);
        }
        if (secret.length < 8) continue; // too short to be a credential after trim

        const replacement = `[REDACTED:${f.ruleId}]`;
        const before = scrubbed;
        scrubbed = scrubbed.split(secret).join(replacement);
        if (scrubbed !== before) {
          redactionCount++;
          perRule.set(f.ruleId, (perRule.get(f.ruleId) || 0) + 1);
        }
      }
      out.write(scrubbed);
      out.write("\n");
    }
    await new Promise<void>((resolve, reject) => {
      out.end((err: Error | null | undefined) => (err ? reject(err) : resolve()));
    });

    if (redactionCount > 0) {
      // Atomic swap — only when something changed. A no-op rename would
      // refresh the transcript mtime and defeat scrub-sweep's watermark,
      // turning incremental sweeps into perpetual full rescans.
      // chmod again: the umask may have stripped bits at create time.
      fs.chmodSync(tmpPath, srcMode);
      fs.renameSync(tmpPath, transcriptPath);
      process.stderr.write(
        `scrub-transcript: ${redactionCount} redaction(s) in ${transcriptPath}\n`,
      );
      appendAudit(transcriptPath, perRule);
    } else {
      try { fs.unlinkSync(tmpPath); } catch { /* ignore */ }
    }
  } catch (err) {
    // Leave original intact; clean up tmp.
    try { fs.unlinkSync(tmpPath); } catch { /* ignore */ }
    process.stderr.write(`scrub-transcript: error: ${(err as Error).message}\n`);
    process.exitCode = 2;
  } finally {
    fs.closeSync(lockFd);
    try { fs.unlinkSync(lockPath); } catch { /* ignore */ }
  }
}

const target = process.argv[2];
if (!target) {
  process.stderr.write("scrub-transcript: missing transcript-path arg\n");
  process.exit(1);
}
scrub(target).catch((e) => {
  process.stderr.write(`scrub-transcript: fatal: ${e.message}\n`);
  process.exit(3);
});
