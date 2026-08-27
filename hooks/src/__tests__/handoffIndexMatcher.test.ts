/**
 * Tests for classifyArtifactPath in handoff-index.ts (issue #283).
 *
 * The PostToolUse hook used to index only handoff files, so plans written to
 * thoughts/shared/plans/ never reached artifact_index.py --file and the
 * PostgreSQL plans table stayed empty. The matcher is a pure function so the
 * routing decision can be tested without spawning anything.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  classifyArtifactPath,
  indexScriptCandidates,
  isIndexableToolEvent,
  isWithinProject,
} from '../handoff-index.js';

describe('indexScriptCandidates', () => {
  it('prefers the centralised OPC dir, then project-relative locations', () => {
    expect(indexScriptCandidates('/some/project', '/Users/me/opc')).toEqual([
      '/Users/me/opc/scripts/core/artifact_index.py',
      '/some/project/scripts/core/artifact_index.py',
      '/some/project/scripts/artifact_index.py',
    ]);
  });

  it('falls back to project-relative locations when no OPC dir is resolvable', () => {
    expect(indexScriptCandidates('/repo', null)).toEqual([
      '/repo/scripts/core/artifact_index.py',
      '/repo/scripts/artifact_index.py',
    ]);
    expect(indexScriptCandidates('/repo')).toEqual(indexScriptCandidates('/repo', null));
  });
});

describe('isWithinProject', () => {
  let root: string;
  let repo: string;
  let outside: string;

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'hoi-'));
    repo = path.join(root, 'repo');
    outside = path.join(root, 'outside');
    fs.mkdirSync(path.join(repo, 'thoughts', 'shared', 'plans'), { recursive: true });
    fs.mkdirSync(outside, { recursive: true });
    fs.writeFileSync(path.join(repo, 'thoughts', 'shared', 'plans', 'x.md'), '# x');
    fs.writeFileSync(path.join(outside, 'secret.md'), '# secret');
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  it('accepts a real file inside the project (trailing slash tolerated)', () => {
    const f = path.join(repo, 'thoughts', 'shared', 'plans', 'x.md');
    expect(isWithinProject(f, repo)).toBe(true);
    expect(isWithinProject(f, repo + path.sep)).toBe(true);
  });

  it('rejects the project root itself, siblings, and traversal', () => {
    expect(isWithinProject(repo, repo)).toBe(false);
    expect(isWithinProject(path.join(outside, 'secret.md'), repo)).toBe(false);
    expect(isWithinProject(path.join(repo, '..', 'outside', 'secret.md'), repo)).toBe(false);
  });

  it('rejects a symlink inside the project that points outside it', () => {
    const link = path.join(repo, 'thoughts', 'shared', 'plans', 'link.md');
    fs.symlinkSync(path.join(outside, 'secret.md'), link);
    expect(isWithinProject(link, repo)).toBe(false);
  });

  it('rejects a symlinked directory that escapes the project', () => {
    const linkDir = path.join(repo, 'thoughts', 'shared', 'plans', 'ext');
    fs.symlinkSync(outside, linkDir);
    expect(isWithinProject(path.join(linkDir, 'secret.md'), repo)).toBe(false);
  });

  it('rejects paths that do not exist', () => {
    expect(isWithinProject(path.join(repo, 'thoughts', 'shared', 'plans', 'nope.md'), repo)).toBe(false);
  });
});

describe('isIndexableToolEvent', () => {
  it('accepts the file-writing tools', () => {
    expect(isIndexableToolEvent('Write')).toBe(true);
    expect(isIndexableToolEvent('Edit')).toBe(true);
    expect(isIndexableToolEvent('MultiEdit')).toBe(true);
  });

  it('rejects everything else', () => {
    expect(isIndexableToolEvent('Read')).toBe(false);
    expect(isIndexableToolEvent('Bash')).toBe(false);
    expect(isIndexableToolEvent('')).toBe(false);
    expect(isIndexableToolEvent(undefined)).toBe(false);
  });
});

describe('classifyArtifactPath', () => {
  it('classifies markdown handoffs', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/handoffs/main/2026-08-27_x.md')).toBe('handoff');
  });

  it('classifies yaml handoffs (.yaml and .yml)', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/handoffs/main/x.yaml')).toBe('handoff');
    expect(classifyArtifactPath('thoughts/shared/handoffs/main/x.yml')).toBe('handoff');
  });

  it('classifies plans under thoughts/shared/plans', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/plans/issue-283.md')).toBe('plan');
    expect(classifyArtifactPath('thoughts/shared/plans/issue-283.md')).toBe('plan');
  });

  it('classifies plans in a subdirectory of plans', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/plans/archive/old.md')).toBe('plan');
  });

  it('ignores non-markdown files in plans', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/plans/notes.yaml')).toBeNull();
    expect(classifyArtifactPath('/repo/thoughts/shared/plans/data.json')).toBeNull();
  });

  it('ignores a "plans" directory outside thoughts/shared', () => {
    expect(classifyArtifactPath('/repo/src/plans/roadmap.md')).toBeNull();
    expect(classifyArtifactPath('/repo/docs/plans.md')).toBeNull();
  });

  it('ignores unrelated markdown and empty paths', () => {
    expect(classifyArtifactPath('/repo/README.md')).toBeNull();
    expect(classifyArtifactPath('')).toBeNull();
  });

  it('prefers handoff when both segments appear', () => {
    expect(classifyArtifactPath('/repo/thoughts/shared/handoffs/plans/x.md')).toBe('handoff');
  });
});
