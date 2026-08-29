/*!
 * Memory injection opt-out.
 *
 * Setting OPC_MEMORY_LOSS=1 in the environment disables the hooks that inject
 * stored learnings into Claude's context (memory-awareness on UserPromptSubmit,
 * session-start-memory-push on SessionStart). Useful for sessions where recall
 * noise is unwanted — demos, benchmarks, or debugging the hooks themselves.
 *
 * Pure: reads only the env object it is handed.
 */

export const MEMORY_LOSS_ENV = 'OPC_MEMORY_LOSS';

const TRUTHY = new Set(['1', 'true', 'yes', 'on']);

export function isMemoryInjectionDisabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const raw = env[MEMORY_LOSS_ENV];
  if (raw === undefined) return false;
  return TRUTHY.has(raw.trim().toLowerCase());
}
