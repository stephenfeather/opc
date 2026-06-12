// Daily sweep: scrub every Claude Code transcript modified since the watermark.
//
// Forward-only policy: on first run, the watermark is initialized to "now",
// so pre-existing transcripts are left untouched. Each successful sweep
// advances the watermark to the start time of the run.
//
// Spawns `dist/scrub-transcript.mjs <path>` per file. The worker is idempotent
// and lock-protected, so racing with SessionEnd hooks is safe.

import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";

const HOME = process.env.HOME || "";
const PROJECTS_DIR = path.join(HOME, ".claude", "projects");
const WATERMARK = path.join(HOME, ".claude", "hooks", "scrub-watermark");
const WORKER = path.join(HOME, ".claude", "hooks", "dist", "scrub-transcript.mjs");

function readWatermark(): number {
  try {
    return parseInt(fs.readFileSync(WATERMARK, "utf8").trim(), 10);
  } catch {
    return NaN;
  }
}

function writeWatermark(ms: number): void {
  fs.writeFileSync(WATERMARK, `${ms}\n`);
}

function listJsonl(dir: string): string[] {
  const out: string[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listJsonl(full));
    else if (e.isFile() && e.name.endsWith(".jsonl")) out.push(full);
  }
  return out;
}

function main(): void {
  const start = Date.now();
  let watermark = readWatermark();
  if (!Number.isFinite(watermark)) {
    // First run — forward-only: skip everything that exists today.
    writeWatermark(start);
    process.stderr.write(
      `scrub-sweep: initialized watermark to ${new Date(start).toISOString()} (forward-only)\n`,
    );
    return;
  }

  if (!fs.existsSync(WORKER)) {
    process.stderr.write(`scrub-sweep: worker missing: ${WORKER}\n`);
    process.exit(1);
  }

  const files = listJsonl(PROJECTS_DIR);
  let scanned = 0;
  let failed = 0;

  for (const f of files) {
    let st: fs.Stats;
    try { st = fs.statSync(f); } catch { continue; }
    if (st.mtimeMs < watermark) continue;
    scanned++;
    const r = spawnSync("node", [WORKER, f], {
      stdio: "inherit",
      env: { ...process.env, SCRUB_TRIGGER: "sweep" },
    });
    if (r.status !== 0) failed++;
  }

  if (failed === 0) writeWatermark(start);
  process.stderr.write(
    `scrub-sweep: scanned ${scanned}, failed ${failed}, elapsed ${Date.now() - start}ms\n`,
  );
  if (failed > 0) process.exit(2);
}

main();
