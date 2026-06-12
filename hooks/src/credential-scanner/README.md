# credential-scanner

Vendored from [coo-quack/sensitive-canary](https://github.com/coo-quack/sensitive-canary) (MIT, develop branch, fetched 2026-05-10).

## What's vendored

| File | Status | Purpose |
|---|---|---|
| `rules.ts` | upstream verbatim | Regex + entropy + luhn rules sourced from gitleaks/TruffleHog |
| `inspector.ts` | upstream verbatim | Scan-and-redact API over the rule set |
| `__tests__/rules.test.ts` | upstream verbatim | Validates each regex against canonical positive/negative samples |
| `__tests__/inspector.test.ts` | upstream verbatim | Integration tests for the inspector API |
| `EXTENSIONS.ts` | local | Custom rules layered on top of vendor rules (OpenRouter, Resend, etc.) |
| `_upstream-*.ts.reference` | upstream, reference only | The original hook wrappers — kept for diffing, not loaded. Our wrappers live in `../../`. |

## Sync policy

Vendor files are kept verbatim so upstream syncs are clean diffs. Local rules go in `EXTENSIONS.ts`.

To pull upstream updates:

```
cd /tmp
gh api repos/coo-quack/sensitive-canary/contents/src/lib/rules.ts -H "Accept: application/vnd.github.raw" > new-rules.ts
diff /tmp/new-rules.ts ~/.dotfiles/claude/hooks/src/credential-scanner/rules.ts
```

## License

Upstream is MIT. See `LICENSE`. Original copyright holder: coo-quack (2026).
