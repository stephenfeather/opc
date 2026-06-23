/**
 * Tests for already-surfaced filtering in memory-awareness.ts (issue #228 item 2).
 *
 * The hook re-surfaces the same top memories every turn. To suppress prior-turn
 * picks BEFORE ranking, it tracks surfaced learning UUIDs per session (in the
 * sessions.surfaced_learning_ids column, keyed by claude_session_id) and passes
 * them to recall_learnings.py as --exclude-ids.
 *
 * Covered:
 *  1. reads surfaced ids and passes them as --exclude-ids with FULL uuids
 *  2. captures full (untruncated) uuids for the union, not the 8-char slice
 *  3. unions returned uuids into the session after recall
 *  4. graceful degradation: no row / null column / DB error -> recall still
 *     runs, no --exclude-ids, hook does not throw
 *  5. cap enforcement bounds the persisted/argv id set
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the DB layer so no real Postgres is touched. readSurfacedIds uses the
// synchronous runPgQuery (needed before recall); persistSurfacedIds uses the
// detached runPgQueryDetached (fire-and-forget, off the prompt hot path).
vi.mock('../shared/db-utils-pg.js', () => ({
  runPgQuery: vi.fn(() => ({ success: true, stdout: '[]', stderr: '' })),
  runPgQueryDetached: vi.fn(() => undefined),
}));

import { runPgQuery, runPgQueryDetached } from '../shared/db-utils-pg.js';
import {
  SURFACED_ID_CAP,
  extractFullIds,
  buildExcludeArgs,
  unionCap,
  readSurfacedIds,
  persistSurfacedIds,
} from '../memory-awareness.js';

const mockedRunPgQuery = vi.mocked(runPgQuery);
const mockedRunPgQueryDetached = vi.mocked(runPgQueryDetached);

const U = (n: number) =>
  `00000000-0000-0000-0000-${String(n).padStart(12, '0')}`;

describe('extractFullIds', () => {
  it('captures full untruncated uuids, not the 8-char display slice', () => {
    const results = [{ id: U(1) }, { id: U(2) }];
    expect(extractFullIds(results)).toEqual([U(1), U(2)]);
    // Each id is a full 36-char uuid, NOT a slice(0,8).
    for (const id of extractFullIds(results)) {
      expect(id.length).toBe(36);
    }
  });

  it('drops missing/empty ids', () => {
    const results = [{ id: U(1) }, {}, { id: '' }, { id: U(2) }];
    expect(extractFullIds(results)).toEqual([U(1), U(2)]);
  });

  it('returns [] for empty/garbage input', () => {
    expect(extractFullIds([])).toEqual([]);
    expect(extractFullIds(undefined as any)).toEqual([]);
  });
});

describe('buildExcludeArgs', () => {
  it('builds --exclude-ids with full uuids', () => {
    expect(buildExcludeArgs([U(1), U(2)])).toEqual(['--exclude-ids', U(1), U(2)]);
  });

  it('returns [] (flag omitted) when the list is empty', () => {
    expect(buildExcludeArgs([])).toEqual([]);
  });
});

describe('unionCap', () => {
  it('unions and dedupes prior + fresh ids', () => {
    expect(unionCap([U(1), U(2)], [U(2), U(3)], 500)).toEqual([U(1), U(2), U(3)]);
  });

  it('bounds the result to the cap (most recent kept)', () => {
    const prior = Array.from({ length: 600 }, (_, i) => U(i));
    const fresh = [U(9001), U(9002)];
    const out = unionCap(prior, fresh, 500);
    expect(out.length).toBe(500);
    // The freshly surfaced ids must survive the cap.
    expect(out).toContain(U(9001));
    expect(out).toContain(U(9002));
  });

  it('handles null/empty prior', () => {
    expect(unionCap([], [U(1)], 500)).toEqual([U(1)]);
    expect(unionCap(null as any, [U(1)], 500)).toEqual([U(1)]);
  });

  it('stays exactly at cap when a full session adds fresh ids (round-1 regression)', () => {
    // Prior already at the cap; new ids arrive. The result must remain exactly
    // SURFACED_ID_CAP, include all fresh ids, and evict the oldest — the bound
    // the appending persist SQL failed to enforce.
    const prior = Array.from({ length: SURFACED_ID_CAP }, (_, i) => U(i));
    const fresh = [U(900), U(901), U(902)];
    const out = unionCap(prior, fresh, SURFACED_ID_CAP);
    expect(out.length).toBe(SURFACED_ID_CAP);
    for (const f of fresh) expect(out).toContain(f);
    expect(out).not.toContain(U(0));
    expect(out).not.toContain(U(1));
    expect(out).not.toContain(U(2));
  });
});

describe('readSurfacedIds', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('reads surfaced ids for the session as full uuids', () => {
    mockedRunPgQuery.mockReturnValue({
      success: true,
      stdout: JSON.stringify([U(1), U(2)]),
      stderr: '',
    });
    expect(readSurfacedIds('sess-abc')).toEqual([U(1), U(2)]);
    // The session id is bound as the query arg.
    const call = mockedRunPgQuery.mock.calls[0];
    expect(call[1]).toContain('sess-abc');
  });

  it('returns [] when there is no session row (null/empty stdout)', () => {
    mockedRunPgQuery.mockReturnValue({ success: true, stdout: '', stderr: '' });
    expect(readSurfacedIds('sess-none')).toEqual([]);
  });

  it('returns [] when the column is NULL (json null)', () => {
    mockedRunPgQuery.mockReturnValue({ success: true, stdout: 'null', stderr: '' });
    expect(readSurfacedIds('sess-null')).toEqual([]);
  });

  it('returns [] and never throws on DB error', () => {
    mockedRunPgQuery.mockReturnValue({ success: false, stdout: '', stderr: 'boom' });
    expect(() => readSurfacedIds('sess-err')).not.toThrow();
    expect(readSurfacedIds('sess-err')).toEqual([]);
  });

  it('returns [] when runPgQuery itself throws', () => {
    mockedRunPgQuery.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    expect(() => readSurfacedIds('sess-throw')).not.toThrow();
    expect(readSurfacedIds('sess-throw')).toEqual([]);
  });

  it('defensively caps an oversized stored row to SURFACED_ID_CAP (newest kept)', () => {
    // A row written before the cap was enforced (or via schema drift) must not
    // produce an unbounded --exclude-ids argv on read.
    const stored = Array.from({ length: SURFACED_ID_CAP + 25 }, (_, i) => U(i));
    mockedRunPgQuery.mockReturnValue({
      success: true,
      stdout: JSON.stringify(stored),
      stderr: '',
    });
    const out = readSurfacedIds('sess-big');
    expect(out.length).toBe(SURFACED_ID_CAP);
    expect(out[out.length - 1]).toBe(U(SURFACED_ID_CAP + 24));
  });
});

describe('persistSurfacedIds', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('runs DETACHED (off the prompt hot path), not synchronously', () => {
    persistSurfacedIds('sess-x', [U(1), U(2)]);
    expect(mockedRunPgQueryDetached).toHaveBeenCalledTimes(1);
    // Persistence is only needed next turn, so it must NOT use the synchronous
    // seam that would add a DB round-trip to the prompt-submit budget.
    expect(mockedRunPgQuery).not.toHaveBeenCalled();
  });

  it('UPSERTS keyed on the PK id (no silent no-op when the row is missing)', () => {
    persistSurfacedIds('sess-x', [U(1), U(2)]);
    const [code, args] = mockedRunPgQueryDetached.mock.calls[0];
    expect(code).toContain('surfaced_learning_ids');
    // Upsert, not a bare UPDATE that no-ops when the SessionStart row is absent.
    expect(code).toContain('INSERT INTO sessions');
    expect(code).toContain('ON CONFLICT (id) DO UPDATE');
    // Replace assignment via EXCLUDED, NOT an append/re-union against the
    // existing column (a `|| $2` + unnest re-union would re-add ids unionCap
    // trimmed and let the array grow without bound — the round-1 finding).
    expect(code).toContain('EXCLUDED.surfaced_learning_ids');
    expect(code).not.toContain('unnest');
    expect(code).not.toContain('||');
    expect(args).toContain('sess-x');
    const payload = JSON.parse((args as string[])[1]);
    expect(payload).toEqual([U(1), U(2)]);
  });

  it('is a no-op when there are no ids', () => {
    persistSurfacedIds('sess-x', []);
    expect(mockedRunPgQueryDetached).not.toHaveBeenCalled();
  });

  it('defensively caps the bound id array to SURFACED_ID_CAP', () => {
    const ids = Array.from({ length: SURFACED_ID_CAP + 50 }, (_, i) => U(i));
    persistSurfacedIds('sess-x', ids);
    const [, args] = mockedRunPgQueryDetached.mock.calls[0];
    // Parse the actual bound array and assert its LENGTH is capped (the old
    // test counted args, not ids, and so never caught an over-cap payload).
    const payload = JSON.parse((args as string[])[1]) as string[];
    expect(payload.length).toBe(SURFACED_ID_CAP);
    // Most-recent tail survives (consistent with unionCap head-trim).
    expect(payload[payload.length - 1]).toBe(U(SURFACED_ID_CAP + 49));
  });

  it('never throws when the detached spawn throws', () => {
    mockedRunPgQueryDetached.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    expect(() => persistSurfacedIds('sess-x', [U(1)])).not.toThrow();
  });
});
