/**
 * Session ID utilities for cross-process coordination.
 *
 * Session IDs are provided by Claude Code on stdin to every hook.
 * This module provides fallback generation for cases where env vars
 * are the only source (e.g., within a single process).
 *
 * The singleton file (.coordination-session-id) was removed in #85
 * because it caused cross-session identity collisions.
 */

/**
 * Generates a new short session ID.
 * Priority: BRAINTRUST_SPAN_ID (first 8 chars) > timestamp-based ID.
 *
 * @returns 8-character session identifier (e.g., "s-m1abc23")
 */
export function generateSessionId(): string {
  const spanId = process.env.BRAINTRUST_SPAN_ID;
  if (spanId) {
    return spanId.slice(0, 8);
  }
  return `s-${Date.now().toString(36)}`;
}

/**
 * Retrieves the session ID for coordination, checking env var sources.
 * Priority: COORDINATION_SESSION_ID env var > BRAINTRUST_SPAN_ID > generated.
 *
 * Hooks should prefer stdin session_id over this function. This exists
 * for contexts where stdin is not available (e.g., session-register
 * setting the env var for child processes).
 *
 * @param options.debug - If true, logs when falling back to generation
 * @returns Session identifier string (e.g., "s-m1abc23")
 */
export function getSessionId(options: { debug?: boolean } = {}): string {
  // First try environment (same process)
  if (process.env.COORDINATION_SESSION_ID) {
    return process.env.COORDINATION_SESSION_ID;
  }

  // Fallback - log if debug enabled
  if (options.debug) {
    console.error(
      '[session-id] WARNING: No COORDINATION_SESSION_ID env var, falling back to BRAINTRUST_SPAN_ID or generating a new one',
    );
  }

  // Fallback to Braintrust span ID or generate new
  return generateSessionId();
}

/**
 * Returns the current project directory path.
 *
 * @returns CLAUDE_PROJECT_DIR env var or current working directory
 */
export function getProject(): string {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}
