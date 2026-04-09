/**
 * Tests for main() handoff-first behavior.
 *
 * These tests verify that:
 * 1. When a handoff for the current workstream has a Ledger section, use it
 *    (not legacy ledger).
 * 2. When the matching handoff lacks a Ledger section, fall back to legacy.
 * 3. When no handoff exists, use legacy ledger.
 * 4. Issue #86 regression: cross-stream handoffs with newer mtime must NOT
 *    be loaded into the current workstream.
 * 5. `source: 'startup'` must not inject handoff content regardless of which
 *    handoffs exist on disk.
 *
 * Each test's tempdir is initialized as a git repo so the workstream resolver
 * in session-start-continuity.ts can find a branch. Tests that expect a
 * specific workstream use `git branch -M <name>` to align the branch with
 * the handoff directory name.
 *
 * Run with: npx vitest run src/__tests__/mainHandoffFirst.test.ts
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('main() handoff-first behavior', () => {
  let testDir: string;
  let originalProjectDir: string | undefined;

  beforeEach(() => {
    // Create a temp directory for each test
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mainHandoffFirst-test-'));

    // Initialize as a git repo so the workstream resolver in the hook finds
    // a branch. Default to `main` — individual tests that need a different
    // branch name call `git branch -M <name>` after creating their handoff
    // dir.
    execSync('git init -b main', { cwd: testDir, stdio: 'ignore' });

    // Save and override CLAUDE_PROJECT_DIR
    originalProjectDir = process.env.CLAUDE_PROJECT_DIR;
    process.env.CLAUDE_PROJECT_DIR = testDir;
  });

  afterEach(() => {
    // Restore original CLAUDE_PROJECT_DIR
    if (originalProjectDir !== undefined) {
      process.env.CLAUDE_PROJECT_DIR = originalProjectDir;
    } else {
      delete process.env.CLAUDE_PROJECT_DIR;
    }

    // Clean up temp directory
    fs.rmSync(testDir, { recursive: true, force: true });
  });

  /**
   * Helper to run the hook with stdin input and capture output
   */
  function runHook(input: object): { stdout: string; stderr: string } {
    const inputJson = JSON.stringify(input);
    const hookPath = path.resolve(__dirname, '../../dist/session-start-continuity.mjs');

    try {
      const stdout = execSync(`echo '${inputJson}' | CLAUDE_PROJECT_DIR="${testDir}" node "${hookPath}"`, {
        encoding: 'utf-8',
        timeout: 5000,
        env: { ...process.env, CLAUDE_PROJECT_DIR: testDir }
      });
      return { stdout, stderr: '' };
    } catch (error: any) {
      return { stdout: error.stdout || '', stderr: error.stderr || '' };
    }
  }

  describe('when handoff has Ledger section', () => {
    it('should use handoff Ledger section instead of legacy ledger', () => {
      // Create handoff with Ledger section
      const sessionName = 'test-session';
      execSync(`git branch -M ${sessionName}`, { cwd: testDir, stdio: 'ignore' });
      const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', sessionName);
      fs.mkdirSync(handoffDir, { recursive: true });

      const handoffContent = `# Work Stream: ${sessionName}

## Ledger
**Updated:** 2025-12-30T12:00:00Z
**Goal:** Handoff goal (NEW)
**Branch:** feature/handoff
**Test:** npm test

### Now
[->] Working from handoff Ledger

### Next
- [ ] Next item from handoff

---

## Context
Detailed context from handoff.
`;
      fs.writeFileSync(path.join(handoffDir, 'current.md'), handoffContent);

      // Also create a legacy ledger (should be IGNORED)
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      const legacyLedgerContent = `# Continuity Ledger: ${sessionName}

## Goal
Legacy ledger goal (OLD - should not be used)

## State
- Done: Nothing
- Now: Legacy focus (OLD)
- Next: Legacy next

## Working Set
- branch: main
`;
      fs.writeFileSync(path.join(ledgerDir, `CONTINUITY_CLAUDE-${sessionName}.md`), legacyLedgerContent);

      // Run hook with resume
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      // Should load from handoff, not legacy ledger
      expect(output.result).toBe('continue');

      // The message or additionalContext should reference handoff content
      const fullOutput = JSON.stringify(output);
      expect(
        fullOutput.includes('Handoff goal') || fullOutput.includes('Working from handoff'),
      ).toBe(true);
      expect(
        !fullOutput.includes('Legacy ledger goal') && !fullOutput.includes('Legacy focus'),
      ).toBe(true);
    });

    it('should return current-workstream handoff even when another stream has a newer one (regression #86)', async () => {
      // This is the regression test for issue #86. Before the fix, the hook
      // scanned all handoff subdirectories, picked the newest file by mtime
      // across streams, and injected it — regardless of which workstream the
      // session belonged to. After the fix, the hook resolves the current
      // workstream from git state and loads ONLY that stream's handoff.
      const currentStream = 'current-work';
      const otherStream = 'other-work';

      // Put the tempdir on the `current-work` branch. The hook's workstream
      // resolver will pick this up and look in `handoffs/current-work/`.
      execSync(`git branch -M ${currentStream}`, { cwd: testDir, stdio: 'ignore' });

      const currentDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', currentStream);
      const otherDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', otherStream);
      fs.mkdirSync(currentDir, { recursive: true });
      fs.mkdirSync(otherDir, { recursive: true });

      // Current stream handoff written FIRST (older mtime).
      const currentHandoff = `# Work Stream: ${currentStream}

## Ledger
**Updated:** 2025-12-29T00:00:00Z
**Goal:** Current stream goal
**Branch:** ${currentStream}

### Now
[->] Current stream focus

---

## Context
Current context.
`;
      fs.writeFileSync(path.join(currentDir, 'current.md'), currentHandoff);

      // Wait for different mtime.
      await new Promise(resolve => setTimeout(resolve, 50));

      // Other stream handoff written LATER (newer mtime). Under the old
      // buggy logic this would be selected because it has the newest mtime
      // across all streams. Under the fix it must be ignored.
      const otherHandoff = `# Work Stream: ${otherStream}

## Ledger
**Updated:** 2025-12-30T12:00:00Z
**Goal:** Other stream goal (WRONG)
**Branch:** ${otherStream}

### Now
[->] Other stream focus (WRONG)

---

## Context
Other context.
`;
      fs.writeFileSync(path.join(otherDir, 'current.md'), otherHandoff);

      // Create legacy ledgers directory (should exist but be fallback)
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      // Run hook with source=resume and a session_id that validates.
      const result = runHook({ source: 'resume', session_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' });
      const output = JSON.parse(result.stdout);
      const fullOutput = JSON.stringify(output);

      // Must load CURRENT stream's content...
      expect(
        fullOutput.includes('Current stream goal') || fullOutput.includes('Current stream focus'),
      ).toBe(true);
      // ...and must NOT load OTHER stream's content, even though its mtime is newer.
      expect(
        !fullOutput.includes('Other stream goal') && !fullOutput.includes('Other stream focus'),
      ).toBe(true);
    });
  });

  describe('when handoff lacks Ledger section', () => {
    it('should fall back to legacy ledger', () => {
      // Create handoff WITHOUT Ledger section
      const sessionName = 'no-ledger-session';
      execSync(`git branch -M ${sessionName}`, { cwd: testDir, stdio: 'ignore' });
      const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', sessionName);
      fs.mkdirSync(handoffDir, { recursive: true });

      const handoffContent = `# Work Stream: ${sessionName}

## Context
This handoff has no Ledger section (old format).

## What Was Done
- Some work
`;
      fs.writeFileSync(path.join(handoffDir, 'task-1.md'), handoffContent);

      // Create legacy ledger (should be used as fallback)
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      const legacyLedgerContent = `# Continuity Ledger: ${sessionName}

## Goal
Fallback legacy goal

## State
- Done: Nothing
- Now: Legacy fallback focus
- Next: Next item

## Working Set
- branch: main
`;
      fs.writeFileSync(path.join(ledgerDir, `CONTINUITY_CLAUDE-${sessionName}.md`), legacyLedgerContent);

      // Run hook
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      // Should fall back to legacy ledger
      const fullOutput = JSON.stringify(output);
      expect(
        fullOutput.includes('Fallback legacy goal') || fullOutput.includes('Legacy fallback focus'),
      ).toBe(true);
    });
  });

  describe('when no handoff exists', () => {
    it('should use legacy ledger', () => {
      // No handoff directory at all
      const sessionName = 'legacy-only-session';

      // Create only legacy ledger
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      const legacyLedgerContent = `# Continuity Ledger: ${sessionName}

## Goal
Pure legacy goal

## State
- Done: Nothing
- Now: Pure legacy focus
- Next: Next

## Working Set
- branch: main
`;
      fs.writeFileSync(path.join(ledgerDir, `CONTINUITY_CLAUDE-${sessionName}.md`), legacyLedgerContent);

      // Run hook
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      // Should use legacy ledger
      const fullOutput = JSON.stringify(output);
      expect(
        fullOutput.includes('Pure legacy goal') || fullOutput.includes('Pure legacy focus'),
      ).toBe(true);
    });

    it('should return continue with no message when no ledger or handoff exists', () => {
      // Empty thoughts directory - no handoffs, no ledgers
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });
      // But no ledger files inside

      // Run hook
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      expect(output.result).toBe('continue');
    });
  });

  describe('handoff directory edge cases', () => {
    it('should handle non-existent handoffs directory gracefully', () => {
      // Create ledger dir but not handoffs dir
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      const legacyLedgerContent = `# Continuity Ledger: test

## Goal
Test goal

## State
- Done: Nothing
- Now: Test focus
- Next: Next

## Working Set
- branch: main
`;
      fs.writeFileSync(path.join(ledgerDir, 'CONTINUITY_CLAUDE-test.md'), legacyLedgerContent);

      // Run hook - should not crash, should use legacy ledger
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      expect(output.result).toBe('continue');
      const fullOutput = JSON.stringify(output);
      expect(
        fullOutput.includes('Test goal') || fullOutput.includes('Test focus'),
      ).toBe(true);
    });

    it('should handle empty handoffs directory', () => {
      // Create empty handoffs directory
      const handoffsDir = path.join(testDir, 'thoughts', 'shared', 'handoffs');
      fs.mkdirSync(handoffsDir, { recursive: true });

      // Create legacy ledger
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });

      const legacyLedgerContent = `# Continuity Ledger: empty-handoffs

## Goal
Empty handoffs test

## State
- Done: Nothing
- Now: Empty handoffs focus
- Next: Next

## Working Set
- branch: main
`;
      fs.writeFileSync(path.join(ledgerDir, 'CONTINUITY_CLAUDE-empty-handoffs.md'), legacyLedgerContent);

      // Run hook
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      expect(output.result).toBe('continue');
    });
  });

  describe('startup vs resume behavior', () => {
    it('should NOT inject handoff content on source=startup even if a matching handoff exists', () => {
      // After issue #86, fresh `startup` sessions always have a brand-new
      // session_id that cannot match any prior handoff by UUID. The hook
      // must stay silent about handoffs on startup rather than guessing
      // based on mtime. The legacy ledger fallback still applies.
      const sessionName = 'startup-test';
      execSync(`git branch -M ${sessionName}`, { cwd: testDir, stdio: 'ignore' });
      const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', sessionName);
      fs.mkdirSync(handoffDir, { recursive: true });

      const handoffContent = `# Work Stream: ${sessionName}

## Ledger
**Updated:** 2025-12-30T12:00:00Z
**Goal:** Startup handoff goal (MUST NOT APPEAR)
**Branch:** ${sessionName}

### Now
[->] Startup handoff focus (MUST NOT APPEAR)

---

## Context
Context details that must not be injected on startup.
`;
      fs.writeFileSync(path.join(handoffDir, 'current.md'), handoffContent);

      // Run hook with startup. No legacy ledger dir, so the fallback is a no-op.
      const result = runHook({ source: 'startup', session_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' });
      const output = JSON.parse(result.stdout);
      const fullOutput = JSON.stringify(output);

      expect(output.result).toBe('continue');
      expect(
        !fullOutput.includes('Startup handoff goal') && !fullOutput.includes('Startup handoff focus'),
      ).toBe(true);
    });

    it('should silently skip handoff lookup when tempdir is not a git repo', () => {
      // resolveWorkstreamName returns null for non-git directories. The hook
      // must silently skip the handoff block and fall through to the legacy
      // ledger path without throwing or injecting the wrong content.
      const sessionName = 'non-git-stream';
      const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', sessionName);
      fs.mkdirSync(handoffDir, { recursive: true });

      const handoffContent = `# Work Stream: ${sessionName}

## Ledger
**Updated:** 2025-12-30T12:00:00Z
**Goal:** Non-git handoff goal (MUST NOT APPEAR)

### Now
[->] Non-git handoff focus (MUST NOT APPEAR)

---

## Context
Context for a handoff that must not be loaded when git resolution fails.
`;
      fs.writeFileSync(path.join(handoffDir, 'current.md'), handoffContent);

      // Remove the .git directory to simulate a non-git project.
      fs.rmSync(path.join(testDir, '.git'), { recursive: true, force: true });

      const result = runHook({ source: 'resume', session_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee' });
      const output = JSON.parse(result.stdout);
      const fullOutput = JSON.stringify(output);

      expect(output.result).toBe('continue');
      expect(
        !fullOutput.includes('Non-git handoff goal') && !fullOutput.includes('Non-git handoff focus'),
      ).toBe(true);
    });

    it('should load full Ledger content on resume/clear/compact', () => {
      // Create handoff with Ledger
      const sessionName = 'resume-test';
      execSync(`git branch -M ${sessionName}`, { cwd: testDir, stdio: 'ignore' });
      const handoffDir = path.join(testDir, 'thoughts', 'shared', 'handoffs', sessionName);
      fs.mkdirSync(handoffDir, { recursive: true });

      const handoffContent = `# Work Stream: ${sessionName}

## Ledger
**Updated:** 2025-12-30T12:00:00Z
**Goal:** Resume test goal with detailed info
**Branch:** feature/resume

### Now
[->] Working on resume functionality

### This Session
- [x] Completed task 1
- [x] Completed task 2

### Next
- [ ] Priority 1
- [ ] Priority 2

### Decisions
- Important decision: reasoning

---

## Context
Full context that should be available on resume.
`;
      fs.writeFileSync(path.join(handoffDir, 'current.md'), handoffContent);

      // Create ledger dir
      const ledgerDir = path.join(testDir, 'thoughts', 'ledgers');
      fs.mkdirSync(ledgerDir, { recursive: true });
      fs.writeFileSync(path.join(ledgerDir, `CONTINUITY_CLAUDE-${sessionName}.md`), '# Ledger\n## Goal\nLegacy');

      // Test resume
      const result = runHook({ source: 'resume', session_id: 'test-123' });
      const output = JSON.parse(result.stdout);

      expect(output.result).toBe('continue');
      // Resume should have more detailed content
      const fullOutput = JSON.stringify(output);
      expect(
        fullOutput.includes('Resume test goal') || fullOutput.includes('Working on resume'),
      ).toBe(true);
    });
  });
});
