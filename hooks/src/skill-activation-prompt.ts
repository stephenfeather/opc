#!/usr/bin/env node
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { spawnSync } from 'child_process';
import { tmpdir } from 'os';

// Import shared resource reader (Phase 4 module)
import { readResourceState, ResourceState } from './shared/resource-reader.js';

// Import validation module for false-positive reduction
import {
    shouldValidateWithLLM,
    buildValidationPrompt,
    SkillMatch,
} from './skill-validation-prompt.js';

interface HookInput {
    session_id: string;
    transcript_path: string;
    cwd: string;
    permission_mode: string;
    prompt: string;
}

// Pattern inference result from Python module
interface PatternInference {
    pattern: string;
    confidence: number;
    signals: string[];
    needs_clarification: boolean;
    clarification_probe: string | null;
    ambiguity_type: string | null;
    alternatives: string[];
    work_breakdown: string;
    work_breakdown_detailed: string;
}

// Pattern-to-agent mapping
const PATTERN_AGENT_MAP: Record<string, string> = {
    'swarm': 'research-agent',
    'hierarchical': 'kraken',
    'pipeline': 'kraken',
    'generator_critic': 'review-agent',
    'adversarial': 'validate-agent',
    'map_reduce': 'kraken',
    'jury': 'validate-agent',
    'blackboard': 'maestro',
    'circuit_breaker': 'kraken',
    'chain_of_responsibility': 'maestro',
    'event_driven': 'kraken',
};

interface PromptTriggers {
    keywords?: string[];
    intentPatterns?: string[];
}

interface SkillRule {
    type: 'guardrail' | 'domain';
    enforcement: 'block' | 'suggest' | 'warn';
    priority: 'critical' | 'high' | 'medium' | 'low';
    promptTriggers?: PromptTriggers;
    description?: string;
}

interface SkillRules {
    version: string;
    skills: Record<string, SkillRule>;
    agents?: Record<string, SkillRule>;
}

interface MatchedSkill {
    name: string;
    matchType: 'keyword' | 'intent';
    matchedTerm?: string;
    config: SkillRule;
    isAgent?: boolean;
    needsValidation?: boolean;
}

/**
 * Run pattern inference using the Python module.
 * Returns null if inference fails or module not available.
 *
 * Cross-platform: Uses spawnSync with cwd option (works on Windows/macOS/Linux).
 */
function runPatternInference(prompt: string, projectDir: string): PatternInference | null {
    try {
        const scriptPath = join(projectDir, 'scripts', 'agentica_patterns', 'pattern_inference.py');
        if (!existsSync(scriptPath)) {
            return null;
        }

        // Build Python code as a string (no shell escaping needed with spawnSync)
        const pythonCode = `
import sys
import json
import importlib.util

# Direct import bypassing __init__.py
spec = importlib.util.spec_from_file_location(
    'pattern_inference',
    ${JSON.stringify(scriptPath)}
)
pattern_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pattern_mod)

prompt = ${JSON.stringify(prompt)}
result = pattern_mod.infer_pattern(prompt)
output = result.to_dict()
output['work_breakdown_detailed'] = pattern_mod.generate_work_breakdown(result)
print(json.dumps(output))
`;

        // Cross-platform: use spawnSync with cwd instead of shell cd && command
        const result = spawnSync('uv', ['run', 'python', '-c', pythonCode], {
            encoding: 'utf-8',
            timeout: 5000,
            cwd: projectDir,
            stdio: ['pipe', 'pipe', 'pipe'],
        });

        if (result.status !== 0 || !result.stdout) {
            return null;
        }

        return JSON.parse(result.stdout.trim()) as PatternInference;
    } catch (err) {
        // Pattern inference is optional - fail silently
        return null;
    }
}

/**
 * Generate agentica orchestration output based on pattern inference.
 */
function generateAgenticaOutput(inference: PatternInference, prompt: string): string {
    let output = '\n';
    output += '='.repeat(50) + '\n';
    output += 'AGENTICA PATTERN INFERENCE\n';
    output += '='.repeat(50) + '\n';
    output += '\n';

    if (inference.confidence >= 0.7) {
        const suggestedAgent = PATTERN_AGENT_MAP[inference.pattern] || 'kraken';
        output += 'SUGGESTED APPROACH:\n';
        output += `  Agent: ${suggestedAgent}\n`;
        output += `  Pattern: ${inference.work_breakdown_detailed}\n`;
        const confidencePct = Math.round(inference.confidence * 100);
        output += `  Confidence: ${confidencePct}%\n`;
        output += '\n';
        output += 'ACTION: Use AskUserQuestion to confirm before spawning:\n';
        output += `  "I'll use ${suggestedAgent} to ${inference.work_breakdown}. Proceed?"\n`;
        output += '  Options: [Yes, proceed] [Different approach] [Let me explain more]\n';
        if (inference.alternatives.length > 0) {
            output += `\nAlternative approaches available: ${inference.alternatives.join(', ')}\n`;
        }
    } else {
        // Low confidence - ask CDM probe
        output += 'CLARIFICATION NEEDED:\n';
        output += '\n';
        if (inference.clarification_probe) {
            output += `Ask the user: "${inference.clarification_probe}"\n`;
        }
        output += '\n';
        output += 'Initial analysis suggests: ' + inference.work_breakdown + '\n';
        const confidencePct = Math.round(inference.confidence * 100);
        output += `Confidence: ${confidencePct}%\n`;
        output += '\n';
        output += 'ACTION: Use AskUserQuestion to clarify before proceeding.\n';
    }

    output += '='.repeat(50) + '\n';
    return output;
}

/**
 * Detect semantic/natural language queries that would benefit from TLDR semantic search.
 * Pattern: Questions starting with how/what/where/why/when/which
 */
function detectSemanticQuery(prompt: string): { isSemanticQuery: boolean; suggestion?: string } {
    // Question word patterns that indicate semantic queries
    const semanticPatterns = [
        /^(how|what|where|why|when|which)\s/i,
        /\?$/,
        /^(find|show|list|get|explain)\s+(all|the|every|any)/i,
        /^.*\s+(implementation|architecture|flow|pattern|logic|system)$/i,
    ];

    const isSemanticQuery = semanticPatterns.some(p => p.test(prompt.trim()));

    if (!isSemanticQuery) {
        return { isSemanticQuery: false };
    }

    // Generate suggestion for semantic search
    const shortPrompt = prompt.length > 50 ? prompt.slice(0, 50) + '...' : prompt;
    const suggestion = `💡 **Semantic Query Detected**

Your question "${shortPrompt}" may benefit from semantic code search.

**Try:**
\`\`\`bash
tldr semantic search "${prompt.slice(0, 100)}" .
\`\`\`

Or use the /explore skill for guided exploration.
`;

    return { isSemanticQuery: true, suggestion };
}

async function main() {
    try {
        // Read input from stdin
        const input = readFileSync(0, 'utf-8');
        let data: HookInput;
        try {
            data = JSON.parse(input);
        } catch {
            // Malformed JSON - exit silently
            process.exit(0);
        }

        // Early validation - prompt is required
        if (!data.prompt || typeof data.prompt !== 'string') {
            process.exit(0);
        }
        const prompt = data.prompt.toLowerCase();

        // Load skill rules (try project first, then global)
        const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
        const homeDir = process.env.HOME || process.env.USERPROFILE || '';
        const projectRulesPath = join(projectDir, '.claude', 'skills', 'skill-rules.json');
        const globalRulesPath = join(homeDir, '.claude', 'skills', 'skill-rules.json');

        let rulesPath = '';
        if (existsSync(projectRulesPath)) {
            rulesPath = projectRulesPath;
        } else if (existsSync(globalRulesPath)) {
            rulesPath = globalRulesPath;
        } else {
            // No rules file found, exit silently
            process.exit(0);
        }
        const rules: SkillRules = JSON.parse(readFileSync(rulesPath, 'utf-8'));

        // Only run pattern inference for complex prompts (20+ words)
        // Short prompts don't need orchestration analysis and this spawns a Python subprocess
        const wordCount = data.prompt.split(/\s+/).length;
        const patternInference = wordCount >= 20 ? runPatternInference(data.prompt, projectDir) : null;

        // Semantic query detection disabled — adds context noise for minimal value
        // Users can invoke /explore directly when needed

        const matchedSkills: MatchedSkill[] = [];

        // Check each skill for matches
        for (const [skillName, config] of Object.entries(rules.skills)) {
            const triggers = config.promptTriggers;
            if (!triggers) {
                continue;
            }

            // Keyword matching
            if (triggers.keywords) {
                const matchedKeyword = triggers.keywords.find(kw =>
                    prompt.includes(kw.toLowerCase())
                );
                if (matchedKeyword) {
                    // Check if this match needs LLM validation
                    const skillMatchForValidation: SkillMatch = {
                        skillName,
                        matchType: 'keyword',
                        matchedTerm: matchedKeyword,
                        prompt: data.prompt, // Use original prompt (not lowercased)
                        skillDescription: config.description,
                        enforcement: config.enforcement,
                    };
                    const needsValidation = shouldValidateWithLLM(skillMatchForValidation);

                    matchedSkills.push({
                        name: skillName,
                        matchType: 'keyword',
                        matchedTerm: matchedKeyword,
                        config,
                        needsValidation,
                    });
                    continue;
                }
            }

            // Intent pattern matching (no validation needed - strong signal)
            if (triggers.intentPatterns) {
                const intentMatch = triggers.intentPatterns.some(pattern => {
                    try {
                        const regex = new RegExp(pattern, 'i');
                        return regex.test(prompt);
                    } catch {
                        // Invalid regex pattern, skip
                        return false;
                    }
                });
                if (intentMatch) {
                    matchedSkills.push({
                        name: skillName,
                        matchType: 'intent',
                        config,
                        needsValidation: false,
                    });
                }
            }
        }

        // Check each agent for matches
        const matchedAgents: MatchedSkill[] = [];
        if (rules.agents) {
            for (const [agentName, config] of Object.entries(rules.agents)) {
                const triggers = config.promptTriggers;
                if (!triggers) {
                    continue;
                }

                // Keyword matching
                if (triggers.keywords) {
                    const matchedKeyword = triggers.keywords.find(kw =>
                        prompt.includes(kw.toLowerCase())
                    );
                    if (matchedKeyword) {
                        // Check if this match needs LLM validation
                        const skillMatchForValidation: SkillMatch = {
                            skillName: agentName,
                            matchType: 'keyword',
                            matchedTerm: matchedKeyword,
                            prompt: data.prompt,
                            skillDescription: config.description,
                            enforcement: config.enforcement,
                        };
                        const needsValidation = shouldValidateWithLLM(skillMatchForValidation);

                        matchedAgents.push({
                            name: agentName,
                            matchType: 'keyword',
                            matchedTerm: matchedKeyword,
                            config,
                            isAgent: true,
                            needsValidation,
                        });
                        continue;
                    }
                }

                // Intent pattern matching (no validation needed - strong signal)
                if (triggers.intentPatterns) {
                    const intentMatch = triggers.intentPatterns.some(pattern => {
                        try {
                            const regex = new RegExp(pattern, 'i');
                            return regex.test(prompt);
                        } catch {
                            // Invalid regex pattern, skip
                            return false;
                        }
                    });
                    if (intentMatch) {
                        matchedAgents.push({
                            name: agentName,
                            matchType: 'intent',
                            config,
                            isAgent: true,
                            needsValidation: false,
                        });
                    }
                }
            }
        }

        // === OUTPUT GENERATION (optimized for minimal context consumption) ===

        // Drop ambiguous matches entirely — if validation is needed, it's not worth the context cost
        const confirmedSkills = matchedSkills.filter(s => !s.needsValidation);
        const confirmedAgents = matchedAgents.filter(a => !a.needsValidation);

        // Only show pattern inference for complex prompts (20+ words, high confidence)
        const showPatternInference = patternInference
            && patternInference.confidence >= 0.7
            && data.prompt.split(/\s+/).length >= 20;

        // Check for blocking skills first (these always pass through)
        const blockingSkills = confirmedSkills.filter(s => s.config.enforcement === 'block');

        // Nothing to show? Exit early.
        if (confirmedSkills.length === 0 && confirmedAgents.length === 0 && !showPatternInference && blockingSkills.length === 0) {
            // Still need to check context warnings below, so don't exit yet
        } else {
            let output = '';

            // Pattern inference: compact format, only for complex prompts
            if (showPatternInference && patternInference) {
                const suggestedAgent = PATTERN_AGENT_MAP[patternInference.pattern] || 'kraken';
                output += `PATTERN: ${patternInference.pattern} → ${suggestedAgent} (${Math.round(patternInference.confidence * 100)}%)\n`;
            }

            // Skill/agent matches: compact format, capped at 5 total
            if (confirmedSkills.length > 0 || confirmedAgents.length > 0) {
                // Sort by priority: critical > high > medium > low, intent matches first within tier
                const priorityOrder: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
                const allConfirmed = [
                    ...confirmedSkills.map(s => ({ ...s, sortKey: priorityOrder[s.config.priority] ?? 3, isAgent: false })),
                    ...confirmedAgents.map(a => ({ ...a, sortKey: priorityOrder[a.config.priority] ?? 3, isAgent: true })),
                ];
                allConfirmed.sort((a, b) => {
                    if (a.sortKey !== b.sortKey) return a.sortKey - b.sortKey;
                    // Intent matches before keyword matches
                    if (a.matchType === 'intent' && b.matchType !== 'intent') return -1;
                    if (b.matchType === 'intent' && a.matchType !== 'intent') return 1;
                    return 0;
                });

                // Cap at 5 suggestions to prevent context floods
                const MAX_SUGGESTIONS = 5;
                const capped = allConfirmed.slice(0, MAX_SUGGESTIONS);
                const skills = capped.filter(s => !s.isAgent);
                const agents = capped.filter(s => s.isAgent);

                if (skills.length > 0) {
                    const hasBlock = skills.some(s => s.config.enforcement === 'block');
                    output += hasBlock ? 'REQUIRED: ' : 'Skills: ';
                    output += skills.map(s => s.name).join(', ') + '\n';
                }
                if (agents.length > 0) {
                    output += 'Agents: ' + agents.map(a => a.name).join(', ') + '\n';
                }
            }

            // Handle blocking enforcement
            if (blockingSkills.length > 0) {
                const blockMessage = `BLOCKING: Invoke ${blockingSkills.map(s => s.name).join(', ')} before responding.\n` + output;
                console.log(JSON.stringify({
                    result: 'block',
                    reason: blockMessage
                }));
                process.exit(0);
            }

            if (output) {
                console.log(output.trimEnd());
            }
        }

        // Check context % from statusLine temp file and add tiered warnings
        // Use hook input session_id first, then env vars as fallback
        // CLAUDE_PPID kept for backwards compatibility with bash wrapper
        const rawSessionId = data.session_id || process.env.CLAUDE_SESSION_ID || process.env.CLAUDE_PPID || 'default';
        const sessionId = rawSessionId.slice(0, 8);  // Match status.py truncation
        const contextFile = join(tmpdir(), `claude-context-pct-${sessionId}.txt`);
        if (existsSync(contextFile)) {
            try {
                const pct = parseInt(readFileSync(contextFile, 'utf-8').trim(), 10);
                let contextWarning = '';

                if (pct >= 90) {
                    contextWarning = '\n' +
                        '='.repeat(50) + '\n' +
                        '  CONTEXT CRITICAL: ' + pct + '%\n' +
                        '  Run /create_handoff NOW before auto-compact!\n' +
                        '='.repeat(50) + '\n';
                } else if (pct >= 80) {
                    contextWarning = '\n' +
                        'CONTEXT WARNING: ' + pct + '%\n' +
                        'Recommend: /create_handoff then /clear soon\n';
                } else if (pct >= 70) {
                    contextWarning = '\nContext at ' + pct + '%. Consider handoff when you reach a stopping point.\n';
                }

                if (contextWarning) {
                    console.log(contextWarning);
                }
            } catch {
                // Ignore read errors
            }
        }

        // Check resource limits and add advisory warnings
        // Phase 5: Soft Limit Advisory
        const resources = readResourceState();
        if (resources && resources.maxAgents > 0) {
            const utilization = resources.activeAgents / resources.maxAgents;
            let resourceWarning = '';

            if (utilization >= 1.0) {
                // At or over limit: CRITICAL
                resourceWarning = '\n' +
                    '='.repeat(50) + '\n' +
                    'RESOURCE CRITICAL: At limit (' + resources.activeAgents + '/' + resources.maxAgents + ' agents)\n' +
                    'Do NOT spawn new agents until existing ones complete.\n' +
                    '='.repeat(50) + '\n';
            } else if (utilization >= 0.8) {
                // Near limit (80%+): WARNING
                const remaining = resources.maxAgents - resources.activeAgents;
                resourceWarning = '\n' +
                    'RESOURCE WARNING: Near limit (' + resources.activeAgents + '/' + resources.maxAgents + ' agents)\n' +
                    'Only ' + remaining + ' agent slot(s) remaining. Limit spawning.\n';
            }

            if (resourceWarning) {
                console.log(resourceWarning);
            }
        }

        process.exit(0);
    } catch (err) {
        console.error('Error in skill-activation-prompt hook:', err);
        process.exit(1);
    }
}

main().catch(err => {
    console.error('Uncaught error:', err);
    process.exit(1);
});
