#!/bin/bash
# Post-Edit Diagnostics Hook - shift-left type/lint validation
exec node "$CLAUDE_PROJECT_DIR/.claude/hooks/dist/post-edit-diagnostics.mjs"
