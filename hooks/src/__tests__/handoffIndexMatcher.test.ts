/**
 * Tests for classifyArtifactPath in handoff-index.ts (issue #283).
 *
 * The PostToolUse hook used to index only handoff files, so plans written to
 * thoughts/shared/plans/ never reached artifact_index.py --file and the
 * PostgreSQL plans table stayed empty. The matcher is a pure function so the
 * routing decision can be tested without spawning anything.
 */

import { describe, it, expect } from 'vitest';
import { classifyArtifactPath } from '../handoff-index.js';

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
