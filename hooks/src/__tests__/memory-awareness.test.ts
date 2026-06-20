/**
 * Unit tests for the pure helpers in memory-awareness.ts (issue #213).
 *
 * Two defects are covered:
 *  1. `sanitizeSearchTerm` must NOT discard discriminating short tokens —
 *     bare numbers ("7") and 2-char tech tokens ("os", "pg", "v2") were being
 *     stripped by a blanket `\b\w{1,2}\b` removal, collapsing a specific prompt
 *     into a generic phrase that matched unrelated cross-project learnings.
 *  2. `isConversationalTurn` must suppress recall on short imperative/meta
 *     replies (e.g. "no, extend it another 7 days") that carry no archival
 *     intent, while letting genuine knowledge queries through.
 */

import { describe, it, expect } from 'vitest';
import { sanitizeSearchTerm, isConversationalTurn } from '../memory-awareness.js';

describe('sanitizeSearchTerm', () => {
  it('preserves bare numbers (regression: "7" was stripped)', () => {
    expect(sanitizeSearchTerm('extend another 7 days')).toBe('extend another 7 days');
  });

  it('preserves 2-char tech tokens like os / pg / v2', () => {
    expect(sanitizeSearchTerm('os import bug')).toBe('os import bug');
    expect(sanitizeSearchTerm('pg pool leak')).toBe('pg pool leak');
    expect(sanitizeSearchTerm('migrate to v2 api')).toBe('migrate to v2 api');
  });

  it('converts underscores and slashes to spaces', () => {
    expect(sanitizeSearchTerm('memory_daemon/db pool')).toBe('memory daemon db pool');
  });

  it('collapses repeated whitespace and trims', () => {
    expect(sanitizeSearchTerm('  multiple   spaces  ')).toBe('multiple spaces');
  });

  it('returns empty string for whitespace-only input', () => {
    expect(sanitizeSearchTerm('   ')).toBe('');
  });
});

describe('isConversationalTurn', () => {
  it('gates the reported prompt: "no, extend it another 7 days"', () => {
    expect(isConversationalTurn('no, extend it another 7 days')).toBe(true);
  });

  it('gates bare affirmations and negations', () => {
    for (const p of ['yes', 'no', 'nope', 'sure', 'ok', 'okay', 'yeah', 'nvm', 'no thanks']) {
      expect(isConversationalTurn(p)).toBe(true);
    }
  });

  it('gates a marker followed by a pronoun-imperative remainder', () => {
    expect(isConversationalTurn('no, do that')).toBe(true);
    expect(isConversationalTurn('yeah do that')).toBe(true);
    expect(isConversationalTurn('yeah, undo that')).toBe(true);
  });

  it('does NOT gate a marker followed by a real query body (review finding #1)', () => {
    // Stripping the lead marker must not discard the substantive remainder.
    expect(isConversationalTurn('no, explain pg pool leak')).toBe(false);
    expect(isConversationalTurn('ok, recall auth pattern')).toBe(false);
    expect(isConversationalTurn('yes, how does reranker work')).toBe(false);
  });

  it('gates short pronoun-led imperatives', () => {
    for (const p of ['do it', 'run it', 'try that', 'undo that', 'redo it', 'extend it another 7 days', 'run it again']) {
      expect(isConversationalTurn(p)).toBe(true);
    }
  });

  it('lets genuine knowledge queries through', () => {
    expect(isConversationalTurn('how does the reranker compute project_match weight')).toBe(false);
    expect(isConversationalTurn('fix the auth bug in session-start')).toBe(false);
    expect(isConversationalTurn('implement the backend resolver for store_learning')).toBe(false);
  });

  it('does NOT gate verb + determiner + noun (not a bare pronoun)', () => {
    expect(isConversationalTurn('fix that bug in the parser')).toBe(false);
    expect(isConversationalTurn('change this function to async')).toBe(false);
    expect(isConversationalTurn('delete that migration file')).toBe(false);
  });

  it('does NOT gate a pronoun-imperative with a memory-bearing tail (review finding #2)', () => {
    // The continuation must consume the whole tail; substantive trailing
    // context after instead/now/etc. means it is not a bare meta turn.
    expect(isConversationalTurn('fix this instead with stored auth pattern')).toBe(false);
    expect(isConversationalTurn('update this now using the pg v2 note')).toBe(false);
  });

  it('gates markers delimited by punctuation other than comma (round-2 finding #1)', () => {
    for (const p of ['no: do that', 'no - do that', 'no. do that', 'ok: run it']) {
      expect(isConversationalTurn(p)).toBe(true);
    }
  });

  it('handles the "another <n> <unit>" quantity grammar (round-2 finding #2)', () => {
    // Numeric quantity tails are meta and gated...
    expect(isConversationalTurn('extend it another 7 days')).toBe(true);
    expect(isConversationalTurn('extend it another 7 business days')).toBe(true);
    // ...but a non-numeric "another <noun phrase>" can be memory-bearing.
    expect(isConversationalTurn('try this another auth pattern')).toBe(false);
    expect(isConversationalTurn('update this another pg pattern')).toBe(false);
  });

  it('lets substantive corrections through even when they start with "no,"', () => {
    expect(
      isConversationalTurn(
        'no, I actually want to understand how the memory daemon extracts thinking blocks across sessions'
      )
    ).toBe(false);
  });

  it('treats empty/whitespace input as conversational (nothing to recall)', () => {
    expect(isConversationalTurn('   ')).toBe(true);
  });
});
