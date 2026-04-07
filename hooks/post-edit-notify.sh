#!/bin/bash
# Post-Edit Notify Hook - notifies TLDR daemon of file changes
exec node "$CLAUDE_PROJECT_DIR/.claude/hooks/dist/post-edit-notify.mjs"
