/*!
 * Tests for the OPC_MEMORY_LOSS opt-out: when set, memory injection hooks
 * (memory-awareness, session-start-memory-push) must not fire.
 */

import { describe, it, expect } from 'vitest';
import { isMemoryInjectionDisabled, MEMORY_LOSS_ENV } from '../shared/memory-opt-out.js';

describe('isMemoryInjectionDisabled', () => {
  it('exports the documented env var name', () => {
    expect(MEMORY_LOSS_ENV).toBe('OPC_MEMORY_LOSS');
  });

  it('returns false when the var is unset', () => {
    expect(isMemoryInjectionDisabled({})).toBe(false);
  });

  it('returns false for empty, "0", and "false"', () => {
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: '' })).toBe(false);
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: '0' })).toBe(false);
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: 'false' })).toBe(false);
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: 'no' })).toBe(false);
  });

  it('returns true for "1"', () => {
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: '1' })).toBe(true);
  });

  it('returns true for truthy words, case-insensitively', () => {
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: 'true' })).toBe(true);
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: 'YES' })).toBe(true);
    expect(isMemoryInjectionDisabled({ OPC_MEMORY_LOSS: ' On ' })).toBe(true);
  });

  it('does not mutate the env object it is given', () => {
    const env = { OPC_MEMORY_LOSS: '1' };
    isMemoryInjectionDisabled(env);
    expect(env).toEqual({ OPC_MEMORY_LOSS: '1' });
  });
});
