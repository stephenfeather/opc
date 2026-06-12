// Local rule extensions. Layered on top of vendor RULES via scan-all.ts.
// Add provider-specific or org-specific patterns here. Keep vendor rules.ts untouched.
//
// Each rule mirrors the shape used by rules.ts (duck-typed; vendor's Rule
// interface is not exported). If upstream changes the shape, update this file
// and scan-all.ts to match.

export interface LocalRule {
  id: string;
  description: string;
  regex: RegExp;
  secretGroup?: number;
  entropyThreshold?: number;
  validate?: (str: string) => boolean;
  category: "secret" | "pii";
}

// Rules covering providers the rentier-digital audit flagged but upstream
// rules.ts does not yet handle: OpenRouter, Resend, Vercel.
//
// Sources:
//   - OpenRouter: https://openrouter.ai/docs (API key prefix `sk-or-`)
//   - Resend:     https://resend.com/docs/api-reference (prefix `re_`)
//   - Vercel:     personal tokens are 24-char hex (no fixed prefix);
//                 detection relies on context, not standalone shape.
//
// Add new rules below. Run `npm test -- credential-scanner` after changes.

export const LOCAL_RULES: LocalRule[] = [
  {
    id: "openrouter-key",
    description: "OpenRouter API Key",
    regex: /\bsk-or-(v1-)?[A-Za-z0-9]{40,}\b/g,
    category: "secret",
  },
  {
    id: "resend-key",
    description: "Resend API Key",
    regex: /\bre_[A-Za-z0-9_]{20,}\b/g,
    entropyThreshold: 3.5,
    category: "secret",
  },
  {
    id: "morph-key",
    description: "Morph API Key (sk- + base64url-ish body, allows _ and -)",
    // Morph keys: `sk-` + ~48 chars of [A-Za-z0-9_-]. Vendor `openai-key`
    // requires [A-Za-z0-9] only, so Morph keys (which contain _ and -) slip
    // through. Exclude prefixes already covered by other rules (or-, proj-,
    // ant-) to avoid double-matching.
    regex: /\bsk-(?!or-|proj-|ant-)[A-Za-z0-9_-]{40,}\b/g,
    entropyThreshold: 3.5,
    category: "secret",
  },
  {
    id: "nia-key",
    description: "Nia API Key (nk_)",
    // Nia keys: `nk_` + 32 alphanumeric chars.
    regex: /\bnk_[A-Za-z0-9]{32}\b/g,
    category: "secret",
  },
  {
    id: "ragie-key",
    description: "Ragie API Key (tnt_)",
    // Ragie tenant keys: `tnt_` + ~55 chars of [A-Za-z0-9_].
    regex: /\btnt_[A-Za-z0-9_]{40,}\b/g,
    entropyThreshold: 3.5,
    category: "secret",
  },
  {
    id: "voyage-key",
    description: "Voyage AI API Key (pa-)",
    // Voyage AI keys: `pa-` + ~43 alphanumeric chars.
    regex: /\bpa-[A-Za-z0-9]{40,}\b/g,
    category: "secret",
  },
  {
    id: "sonarqube-token",
    description: "SonarQube/SonarCloud Token (squ_/sqa_/sqp_ + 40 hex)",
    // Modern Sonar tokens carry a 3-letter prefix: squ_ (user), sqa_ (analysis),
    // sqp_ (project). Legacy bare 40-hex tokens look identical to git SHAs and
    // are caught only via context rules (SONAR_TOKEN= / SONARQUBE_TOKEN=).
    regex: /\bsq[uap]_[a-f0-9]{40}\b/g,
    category: "secret",
  },
  {
    id: "sonarqube-token-ctx",
    description: "SonarQube token by env-var context (SONAR_TOKEN/SONARQUBE_TOKEN)",
    // Catches legacy bare-hex tokens by their assignment context.
    regex: /\bSONAR(?:QUBE)?_TOKEN\s*[:=]\s*["']?([A-Za-z0-9]{32,64})["']?/g,
    secretGroup: 1,
    category: "secret",
  },
  {
    id: "arize-key",
    description: "Arize AI API Key (ak- + UUID + suffix)",
    // Arize keys: `ak-` + UUID + `-` + base64url-ish suffix.
    regex: /\bak-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[A-Za-z0-9_-]{20,}\b/g,
    category: "secret",
  },
  {
    id: "digitalocean-pat",
    description: "DigitalOcean Personal Access Token (dop_v1_)",
    // Format: `dop_v1_` + 64 lowercase hex chars. Introduced ~2023.
    // Older unprefixed 64-char hex tokens are not detectable without
    // unacceptable false-positive rates — same issue as Vercel tokens.
    regex: /\bdop_v1_[a-f0-9]{64}\b/g,
    category: "secret",
  },
  {
    id: "url-basic-auth",
    description: "URL with embedded basic-auth credentials (any scheme)",
    // Generic `<scheme>://<user>:<pass>@` — covers https, http, ftp, ssh,
    // git, redis (with empty user), and any other RFC-3986-style scheme.
    // Vendor `connection-string` already covers mongodb/mysql/postgres/redis
    // when the username is non-empty; overlapping matches converge to the
    // same redaction on second pass thanks to the marker-skip guard.
    regex: /\b[a-zA-Z][a-zA-Z0-9+.\-]*:\/\/[^:/\s@]*:[^@\s/]+@/g,
    category: "secret",
  },
];

// Vendor rules disabled for the transcript-scrubber pipeline. The backfill
// against ~/.claude/projects (2026-05-11) revealed these rules produce more
// false positives than signal in conversational transcripts:
//   - pii-phone-jp: matches numeric UUID segments like `0XXX-YYYY-ZZZZ` and
//     corrupted ~91 UUIDs across 332 historical transcripts.
//   - pii-email:   captures git SSH URLs (`git@github.com:user/repo`) and
//     any email in docs/comments — far broader than credential intent.
//   - pii-ipv4:    RFC1918-only, but private IPs in test fixtures and log
//     dumps are not secrets and redacting them damages reproducibility.
// Credential-leak defense (the actual goal) is unaffected.
export const DISABLED_VENDOR_RULES: Set<string> = new Set([
  "pii-phone-jp",
  "pii-email",
  "pii-ipv4",
]);
