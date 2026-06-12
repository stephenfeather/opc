// src/prompt-secret-block.ts
import { readFileSync } from "fs";

// src/credential-scanner/rules.ts
function luhn(str) {
  const digits = str.replace(/\D/g, "");
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = parseInt(digits[i] ?? "", 10);
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}
function entropy(str) {
  if (str.length === 0) return 0;
  const freq = {};
  for (const ch of str) freq[ch] = (freq[ch] ?? 0) + 1;
  let h = 0;
  const n = str.length;
  for (const count of Object.values(freq)) {
    const p = count / n;
    h -= p * Math.log2(p);
  }
  return h;
}
var SECRET_RULES = [
  // Cloud
  {
    id: "aws-access-key",
    description: "AWS Access Key ID",
    regex: /\b(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}\b/g,
    category: "secret"
  },
  {
    id: "gcp-api-key",
    description: "Google Cloud API Key",
    regex: /AIza[0-9A-Za-z_-]{35}/g,
    category: "secret"
  },
  {
    id: "private-key",
    description: "PEM Private Key",
    // Covers RSA, EC, DSA, PGP, and OpenSSH private keys. Prefer the full
    // BEGIN..END block (lazy) so scrubbing removes the key material, not
    // just the header; fall back to the bare header when END is absent.
    regex: /(?:-----BEGIN (?:RSA |EC |DSA |PGP |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |PGP |OPENSSH )?PRIVATE KEY-----)|(?:-----BEGIN (?:RSA |EC |DSA |PGP |OPENSSH )?PRIVATE KEY)/g,
    category: "secret"
  },
  // Source control
  {
    id: "github-pat",
    description: "GitHub Personal Access Token",
    regex: /gh[pousr]_[A-Za-z0-9]{36,255}/g,
    category: "secret"
  },
  {
    id: "github-fine-grained",
    description: "GitHub Fine-Grained Token",
    regex: /github_pat_[A-Za-z0-9_]{82}/g,
    category: "secret"
  },
  {
    id: "gitlab-pat",
    description: "GitLab Personal Access Token",
    regex: /glpat-[A-Za-z0-9_=-]{20,22}/g,
    category: "secret"
  },
  // Package registries
  {
    id: "npm-token",
    description: "npm Access Token",
    regex: /npm_[A-Za-z0-9]{36}/g,
    category: "secret"
  },
  // Communication
  {
    id: "slack-token",
    description: "Slack Token",
    regex: /xox[baprs]-[0-9a-zA-Z-]{10,72}/g,
    category: "secret"
  },
  {
    id: "slack-webhook",
    description: "Slack Webhook URL",
    regex: /https:\/\/hooks\.slack\.com\/services\/T[A-Za-z0-9_]{8,10}\/B[A-Za-z0-9_]{8,12}\/[A-Za-z0-9_]{23,24}/g,
    category: "secret"
  },
  {
    id: "discord-webhook",
    description: "Discord Webhook URL",
    regex: /https:\/\/discord(?:app)?\.com\/api\/webhooks\/[0-9]{17,20}\/[A-Za-z0-9_-]{68}/g,
    category: "secret"
  },
  {
    id: "telegram-bot-token",
    description: "Telegram Bot Token",
    regex: /[0-9]{8,10}:AA[0-9A-Za-z_-]{33}/g,
    category: "secret"
  },
  {
    id: "twilio-sid",
    description: "Twilio Account SID",
    regex: /AC[0-9a-f]{32}/g,
    category: "secret"
  },
  // Email services
  {
    id: "sendgrid-key",
    description: "SendGrid API Key",
    regex: /SG\.[A-Za-z0-9_-]{20,24}\.[A-Za-z0-9_-]{39,50}/g,
    category: "secret"
  },
  {
    id: "mailgun-key",
    description: "Mailgun API Key",
    regex: /key-[0-9a-zA-Z]{32}/g,
    category: "secret"
  },
  {
    id: "mailchimp-key",
    description: "Mailchimp API Key",
    regex: /[0-9a-f]{32}-us[0-9]{1,2}/g,
    category: "secret"
  },
  // Payment
  {
    id: "stripe-secret-key",
    description: "Stripe Secret Key",
    regex: /sk_(live|test)_[0-9a-zA-Z]{24}/g,
    category: "secret"
  },
  {
    id: "stripe-restricted-key",
    description: "Stripe Restricted Key",
    regex: /rk_(live|test)_[0-9a-zA-Z]{24}/g,
    category: "secret"
  },
  // AI services
  {
    id: "openai-key",
    description: "OpenAI API Key (legacy)",
    regex: /sk-(?!proj-|ant-)[A-Za-z0-9]{48}/g,
    category: "secret"
  },
  {
    id: "openai-project-key",
    description: "OpenAI Project API Key",
    regex: /sk-proj-[A-Za-z0-9_-]{40,}/g,
    entropyThreshold: 3.5,
    category: "secret"
  },
  {
    id: "anthropic-key",
    description: "Anthropic API Key",
    regex: /sk-ant-[A-Za-z0-9_-]{95}/g,
    category: "secret"
  },
  // Auth tokens
  {
    id: "jwt",
    description: "JSON Web Token (JWT)",
    regex: /eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g,
    category: "secret"
  },
  // Generic / env-based
  {
    id: "generic-secret",
    description: "Generic API Key / Secret",
    regex: /(api[_-]?key|secret[_-]?key|access[_-]?token|api[_-]?secret)\s*[:=]\s*['"]?([A-Za-z0-9\-_.]{20,})/gi,
    secretGroup: 2,
    entropyThreshold: 3.5,
    category: "secret"
  },
  {
    id: "env-assignment",
    description: ".env style secret assignment",
    regex: /\b[A-Z_]*(SECRET|PASSWORD|PASSWD|TOKEN|API_KEY|PRIVATE_KEY)[A-Z_0-9]*\s*=\s*(\S{8,})/g,
    secretGroup: 2,
    entropyThreshold: 3,
    category: "secret"
  },
  {
    id: "connection-string",
    description: "Database Connection String with credentials",
    regex: /(mongodb|mysql|postgres|postgresql|redis):\/\/[^:\s]+:[^@\s]+@/g,
    category: "secret"
  }
];
var PII_RULES = [
  {
    id: "pii-email",
    description: "Email Address",
    regex: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
    category: "pii"
  },
  {
    id: "pii-credit-card",
    description: "Credit Card Number",
    // Visa (16d) | Mastercard (16d) | Amex (15d) | Discover (16d)
    // Optional spaces or dashes between digit groups
    regex: /\b(?:4[0-9]{3}(?:[\s-]?[0-9]{4}){3}|5[1-5][0-9]{2}(?:[\s-]?[0-9]{4}){3}|3[47][0-9]{2}[\s-]?[0-9]{6}[\s-]?[0-9]{5}|6(?:011|5[0-9]{2})[0-9](?:[\s-]?[0-9]{4}){3})\b/g,
    validate: luhn,
    category: "pii"
  },
  {
    id: "pii-ssn",
    description: "US Social Security Number",
    regex: /\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b/g,
    category: "pii"
  },
  {
    id: "pii-phone-us",
    description: "US Phone Number",
    regex: /\b(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b/g,
    category: "pii"
  },
  {
    id: "pii-phone-jp",
    description: "Japanese Phone Number",
    regex: /\b0\d{1,4}[\s-]\d{1,4}[\s-]\d{4}\b/g,
    category: "pii"
  },
  {
    id: "pii-postal-jp",
    description: "Japanese Postal Code",
    // Require 〒 prefix to avoid false positives (e.g. phone number fragments)
    regex: /〒\d{3}[\s-]\d{4}/g,
    category: "pii"
  },
  {
    id: "pii-ipv4",
    description: "IPv4 Address (private range)",
    // Only flag RFC-1918 private addresses to reduce noise
    regex: /\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b/g,
    category: "pii"
  }
];
var RULES = [...SECRET_RULES, ...PII_RULES];
function redact(str) {
  if (str.length <= 8) return "****";
  return `${str.slice(0, 4)}****${str.slice(-4)}`;
}

// src/credential-scanner/EXTENSIONS.ts
var LOCAL_RULES = [
  {
    id: "openrouter-key",
    description: "OpenRouter API Key",
    regex: /\bsk-or-(v1-)?[A-Za-z0-9]{40,}\b/g,
    category: "secret"
  },
  {
    id: "resend-key",
    description: "Resend API Key",
    regex: /\bre_[A-Za-z0-9_]{20,}\b/g,
    entropyThreshold: 3.5,
    category: "secret"
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
    category: "secret"
  },
  {
    id: "nia-key",
    description: "Nia API Key (nk_)",
    // Nia keys: `nk_` + 32 alphanumeric chars.
    regex: /\bnk_[A-Za-z0-9]{32}\b/g,
    category: "secret"
  },
  {
    id: "ragie-key",
    description: "Ragie API Key (tnt_)",
    // Ragie tenant keys: `tnt_` + ~55 chars of [A-Za-z0-9_].
    regex: /\btnt_[A-Za-z0-9_]{40,}\b/g,
    entropyThreshold: 3.5,
    category: "secret"
  },
  {
    id: "voyage-key",
    description: "Voyage AI API Key (pa-)",
    // Voyage AI keys: `pa-` + ~43 alphanumeric chars.
    regex: /\bpa-[A-Za-z0-9]{40,}\b/g,
    category: "secret"
  },
  {
    id: "sonarqube-token",
    description: "SonarQube/SonarCloud Token (squ_/sqa_/sqp_ + 40 hex)",
    // Modern Sonar tokens carry a 3-letter prefix: squ_ (user), sqa_ (analysis),
    // sqp_ (project). Legacy bare 40-hex tokens look identical to git SHAs and
    // are caught only via context rules (SONAR_TOKEN= / SONARQUBE_TOKEN=).
    regex: /\bsq[uap]_[a-f0-9]{40}\b/g,
    category: "secret"
  },
  {
    id: "sonarqube-token-ctx",
    description: "SonarQube token by env-var context (SONAR_TOKEN/SONARQUBE_TOKEN)",
    // Catches legacy bare-hex tokens by their assignment context.
    regex: /\bSONAR(?:QUBE)?_TOKEN\s*[:=]\s*["']?([A-Za-z0-9]{32,64})["']?/g,
    secretGroup: 1,
    category: "secret"
  },
  {
    id: "arize-key",
    description: "Arize AI API Key (ak- + UUID + suffix)",
    // Arize keys: `ak-` + UUID + `-` + base64url-ish suffix.
    regex: /\bak-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}-[A-Za-z0-9_-]{20,}\b/g,
    category: "secret"
  },
  {
    id: "digitalocean-pat",
    description: "DigitalOcean Personal Access Token (dop_v1_)",
    // Format: `dop_v1_` + 64 lowercase hex chars. Introduced ~2023.
    // Older unprefixed 64-char hex tokens are not detectable without
    // unacceptable false-positive rates — same issue as Vercel tokens.
    regex: /\bdop_v1_[a-f0-9]{64}\b/g,
    category: "secret"
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
    category: "secret"
  }
];
var DISABLED_VENDOR_RULES = /* @__PURE__ */ new Set([
  "pii-phone-jp",
  "pii-email",
  "pii-ipv4"
]);

// src/credential-scanner/scan-all.ts
function scanAll(text) {
  const findings = [];
  const activeVendor = RULES.filter((r) => !DISABLED_VENDOR_RULES.has(r.id));
  const allRules = [...activeVendor, ...LOCAL_RULES];
  for (const rule of allRules) {
    for (const match of text.matchAll(rule.regex)) {
      const secretValue = rule.secretGroup != null ? match[rule.secretGroup] : match[0];
      if (!secretValue) continue;
      if (rule.entropyThreshold != null && entropy(secretValue) < rule.entropyThreshold)
        continue;
      if (rule.validate != null && !rule.validate(match[0])) continue;
      findings.push({
        ruleId: rule.id,
        description: rule.description,
        category: rule.category,
        matchRedacted: redact(secretValue),
        secretValue
      });
    }
  }
  return findings;
}

// src/prompt-secret-block.ts
/*!
 * Prompt Secret Block — UserPromptSubmit
 *
 * Scans the user's prompt with scanAll() (vendor RULES + LOCAL_RULES) and
 * hard-blocks submission if any credential-category finding is detected.
 * The prompt never enters the Claude Code transcript when blocked.
 *
 * Scope: blocks only `category: "secret"` findings. PII findings (names,
 * emails, phone numbers) from the vendor PII rules would produce too many
 * false positives on legitimate prompts — that's a separate guardrail with
 * different ergonomics. Add a `prompt-pii-block.ts` if PII blocking is
 * later desired.
 *
 * Upstream sensitive-canary supports inline allow tags like [allow-secret];
 * we omit that machinery here. If a legitimate secret-shaped value needs
 * to be sent (rare), the user can rephrase or paste it from a tool call
 * after the secret has been used. Keeping the surface small reduces the
 * chance the bypass is misused.
 */
function readStdin() {
  return readFileSync(0, "utf-8");
}
function main() {
  let input;
  try {
    input = JSON.parse(readStdin());
  } catch {
    process.exit(0);
  }
  const prompt = input.prompt ?? "";
  if (!prompt) process.exit(0);
  const findings = scanAll(prompt).filter(
    (f) => f.category === "secret"
  );
  if (findings.length === 0) process.exit(0);
  const seen = /* @__PURE__ */ new Set();
  const unique = findings.filter((f) => {
    const key = `${f.ruleId}:${f.matchRedacted}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const lines = unique.map(
    (f) => `  \u2022 ${f.description} (${f.ruleId}): ${f.matchRedacted}`
  );
  const reason = "Your prompt contains values that look like live credentials. Blocking submission to keep them out of the Claude Code transcript.\n\nDetected:\n" + lines.join("\n") + "\n\nWhat to do:\n  1. Remove or redact the values above and resubmit.\n  2. If a redacted excerpt is enough, replace the secret with [REDACTED] and describe its role.\n  3. If this is a false positive (e.g. an example value from public docs), rephrase so the value is wrapped in obvious markers and resubmit.";
  const output = { decision: "block", reason };
  process.stdout.write(JSON.stringify(output) + "\n");
  process.exit(0);
}
main();
