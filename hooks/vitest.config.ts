import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'src/__tests__/**/*.test.ts',
      'src/**/__tests__/**/*.test.ts',
    ],
    exclude: ['**/node_modules/**'],
    env: {
      // Issue #291: hook modules that read stdin at import time must fail
      // instantly under test (as readFileSync(0) used to) rather than wait out
      // the EAGAIN idle cap on vitest's never-closing stdin.
      HOOK_STDIN_MAX_IDLE_MS: '0',
    },
  },
});
