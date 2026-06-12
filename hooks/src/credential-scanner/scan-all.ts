// Combined scanner: vendor RULES + LOCAL_RULES, no patches to rules.ts.
//
// Use this instead of vendor `scan()` when you want both rule sets.

import { RULES, redact, entropy, type Finding } from "./rules.ts";
import { LOCAL_RULES, DISABLED_VENDOR_RULES, type LocalRule } from "./EXTENSIONS.ts";

type AnyRule = (typeof RULES)[number] | LocalRule;

export function scanAll(text: string): Finding[] {
  const findings: Finding[] = [];
  const activeVendor = RULES.filter((r) => !DISABLED_VENDOR_RULES.has(r.id));
  const allRules: AnyRule[] = [...activeVendor, ...LOCAL_RULES];

  for (const rule of allRules) {
    for (const match of text.matchAll(rule.regex)) {
      const secretValue =
        rule.secretGroup != null ? match[rule.secretGroup] : match[0];
      if (!secretValue) continue;
      if (
        rule.entropyThreshold != null &&
        entropy(secretValue) < rule.entropyThreshold
      )
        continue;
      if (rule.validate != null && !rule.validate(match[0])) continue;

      findings.push({
        ruleId: rule.id,
        description: rule.description,
        category: rule.category,
        matchRedacted: redact(secretValue),
        secretValue,
      });
    }
  }

  return findings;
}
