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

  it('returns [] (flag omitted) when there is no session id', () => {
    expect(surfacedSessionArgs(undefined)).toEqual([]);
    expect(surfacedSessionArgs('')).toEqual([]);
  });
});
