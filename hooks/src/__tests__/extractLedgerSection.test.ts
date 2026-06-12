/**
 * Tests for extractLedgerSection() function
 *
 * Phase 2a TDD: These tests are written BEFORE the implementation.
 * They should FAIL until the function is implemented.
 *
 * Ported from node:test to vitest (issue #156).
 */

import { describe, it, expect } from 'vitest';

// Import the function under test
import { extractLedgerSection } from '../session-start-continuity.js';

describe('extractLedgerSection', () => {
  it('should return correct Ledger section from valid handoff', () => {
    const handoffContent = `# Work Stream: test-session

## Ledger
**Updated:** 2025-12-30T00:00:00Z
**Goal:** Test the new format
**Branch:** main
**Test:** npm test

### Now
[->] Testing new format

### This Session
- [x] Completed item 1
- [x] Completed item 2

### Next
- [ ] Priority 1
- [ ] Priority 2

### Decisions
- Decision 1: Rationale

---

## Context
Detailed context here that should NOT be included.
More context lines.
`;

    const result = extractLedgerSection(handoffContent);

    expect(result).not.toBeNull();
    expect(result!.startsWith('## Ledger')).toBe(true);
    expect(result!.includes('**Goal:** Test the new format')).toBe(true);
    expect(result!.includes('### Now')).toBe(true);
    expect(result!.includes('[->] Testing new format')).toBe(true);
    expect(result!.includes('### Decisions')).toBe(true);
    expect(result!.includes('## Context')).toBe(false);
    expect(result!.includes('Detailed context here')).toBe(false);
  });

  it('should return null for handoff without Ledger section', () => {
    const handoffContent = `# Work Stream: test-session

## Context
This handoff has no Ledger section.
Just context directly.

## What Was Done
- Some work
`;

    const result = extractLedgerSection(handoffContent);

    expect(result).toBeNull();
  });

  it('should return null for empty file', () => {
    const handoffContent = '';

    const result = extractLedgerSection(handoffContent);

    expect(result).toBeNull();
  });

  it('should handle Ledger section at end of file (no --- separator)', () => {
    const handoffContent = `# Work Stream: test-session

## Ledger
**Updated:** 2025-12-30T00:00:00Z
**Goal:** Test edge case
**Branch:** feature/test

### Now
[->] Current task

### Next
- [ ] Future task`;

    const result = extractLedgerSection(handoffContent);

    expect(result).not.toBeNull();
    expect(result!.startsWith('## Ledger')).toBe(true);
    expect(result!.includes('**Goal:** Test edge case')).toBe(true);
    expect(result!.includes('### Now')).toBe(true);
    expect(result!.includes('### Next')).toBe(true);
    expect(result!.includes('Future task')).toBe(true);
  });

  it('should handle multiple ## headings after Ledger - stops at first ---', () => {
    const handoffContent = `# Work Stream: test-session

## Ledger
**Updated:** 2025-12-30T00:00:00Z
**Goal:** Multiple headings test
**Branch:** main

### Now
[->] Current focus

### Decisions
- Key decision

---

## Context
Context section.

## What Was Done
Work section.

## Blockers
Blockers section.
`;

    const result = extractLedgerSection(handoffContent);

    expect(result).not.toBeNull();
    expect(result!.startsWith('## Ledger')).toBe(true);
    expect(result!.includes('### Decisions')).toBe(true);
    expect(result!.includes('## Context')).toBe(false);
    expect(result!.includes('## What Was Done')).toBe(false);
    expect(result!.includes('## Blockers')).toBe(false);
  });

  it('should handle Ledger with no --- but next ## heading', () => {
    // Edge case: No --- separator, but there's a ## heading that ends the Ledger
    const handoffContent = `# Work Stream: test-session

## Ledger
**Updated:** 2025-12-30T00:00:00Z
**Goal:** No separator test

### Now
[->] Working

## Context
This is after Ledger, should not be included.
`;

    const result = extractLedgerSection(handoffContent);

    expect(result).not.toBeNull();
    expect(result!.includes('**Goal:** No separator test')).toBe(true);
    expect(result!.includes('### Now')).toBe(true);
    expect(result!.includes('## Context')).toBe(false);
  });

  it('should trim whitespace from extracted content', () => {
    const handoffContent = `# Work Stream: test-session

## Ledger

**Updated:** 2025-12-30T00:00:00Z


### Now
[->] Task


---

## Context
`;

    const result = extractLedgerSection(handoffContent);

    expect(result).not.toBeNull();
    // The result should be trimmed (no leading/trailing whitespace in content)
    expect(result!.endsWith('\n\n\n')).toBe(false);
  });
});
