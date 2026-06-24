/**
 * Tests for already-surfaced filtering (issue #228 item 2), hook side.
 *
 * The read / exclude / persist logic lives in recall_learnings.py (driven by
 * --surfaced-session) so the hook pays only one Python/uv process per prompt.
 * The hook's remaining responsibility is to pass the session id through to
 * recall, covered here. The server-side behaviour is covered by the Python
 * suite (tests/test_recall_event_logging.py).
 */

import { describe, it, expect } from 'vitest';

import { surfacedSessionArgs } from '../memory-awareness.js';

describe('surfacedSessionArgs', () => {
  it('builds --surfaced-session with the session id', () => {
    expect(surfacedSessionArgs('sess-abc')).toEqual(['--surfaced-session', 'sess-abc']);
  });

  it('passes a real UUID session id (matches SAFE_ID_PATTERN)', () => {
    const uuid = '11111111-2222-3333-4444-555555555555';
    expect(surfacedSessionArgs(uuid)).toEqual(['--surfaced-session', uuid]);
  });

  it('returns [] (flag omitted) when there is no session id', () => {
    expect(surfacedSessionArgs(undefined)).toEqual([]);
    expect(surfacedSessionArgs('')).toEqual([]);
  });

  it('returns [] for an unsafe session id with path/shell metachars (defense-in-depth)', () => {
    // SAFE_ID_PATTERN = ^[a-zA-Z0-9_-]{1,64}$ — rejects path/shell metachars.
    // (Dashes are allowed: real UUIDs contain them; a dash-led value is caught
    // downstream by argparse, per the security review.)
    expect(surfacedSessionArgs('../etc/passwd')).toEqual([]);
    expect(surfacedSessionArgs('$(whoami)')).toEqual([]);
    expect(surfacedSessionArgs('a'.repeat(65))).toEqual([]);
  });
});
