import { describe, it, expect } from "vitest";
import {
  processInput,
  isSubagentTypeValid,
  suggestAgents,
  buildDenyResponse,
  type HookInput,
} from "../agent-subagent-guard";

describe("agent-subagent-guard", () => {
  describe("isSubagentTypeValid", () => {
    it("returns true for non-empty string", () => {
      expect(isSubagentTypeValid("scout")).toBe(true);
    });

    it("returns false for null", () => {
      expect(isSubagentTypeValid(null)).toBe(false);
    });

    it("returns false for undefined", () => {
      expect(isSubagentTypeValid(undefined)).toBe(false);
    });

    it("returns false for empty string", () => {
      expect(isSubagentTypeValid("")).toBe(false);
    });

    it("returns false for whitespace-only string", () => {
      expect(isSubagentTypeValid("  ")).toBe(false);
    });

    it("returns false for general-purpose", () => {
      expect(isSubagentTypeValid("general-purpose")).toBe(false);
    });

    it("returns false for general-purpose with mixed case", () => {
      expect(isSubagentTypeValid("General-Purpose")).toBe(false);
    });
  });

  describe("suggestAgents", () => {
    it("suggests sleuth for bug-related prompts", () => {
      const result = suggestAgents("Debug this failing test");
      expect(result.some((s) => s.includes("sleuth"))).toBe(true);
    });

    it("suggests kraken for implementation prompts", () => {
      const result = suggestAgents("Implement the new authentication feature");
      expect(result.some((s) => s.includes("kraken"))).toBe(true);
    });

    it("suggests scout for search prompts", () => {
      const result = suggestAgents("Search the codebase for patterns");
      expect(result.some((s) => s.includes("scout"))).toBe(true);
    });

    it("suggests aegis for security prompts", () => {
      const result = suggestAgents("Run a security audit on the API");
      expect(result.some((s) => s.includes("aegis"))).toBe(true);
    });

    it("returns default suggestions when no keywords match", () => {
      const result = suggestAgents("xyzzy foobar baz");
      expect(result.some((s) => s.includes("scout"))).toBe(true);
      expect(result.some((s) => s.includes("kraken"))).toBe(true);
      expect(result.some((s) => s.includes("spark"))).toBe(true);
    });

    it("limits suggestions to 3", () => {
      // Prompt that matches many keywords
      const result = suggestAgents("debug and fix the broken build, then test and review");
      expect(result.length).toBeLessThanOrEqual(3);
    });
  });

  describe("processInput", () => {
    it("allows Agent calls with explicit subagent_type", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: "scout", prompt: "Find files" },
      };
      expect(processInput(input)).toEqual({});
    });

    it("denies Agent calls with null subagent_type", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: null, prompt: "Search the codebase" },
      };
      const result = processInput(input);
      expect(result).toHaveProperty("hookSpecificOutput.permissionDecision", "deny");
    });

    it("denies Agent calls with missing subagent_type", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { prompt: "Implement a feature" },
      };
      const result = processInput(input);
      expect(result).toHaveProperty("hookSpecificOutput.permissionDecision", "deny");
    });

    it("denies Agent calls with empty subagent_type", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: "  ", prompt: "Do something" },
      };
      const result = processInput(input);
      expect(result).toHaveProperty("hookSpecificOutput.permissionDecision", "deny");
    });

    it("denies Agent calls with explicit general-purpose subagent_type", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: "general-purpose", prompt: "Do something" },
      };
      const result = processInput(input);
      expect(result).toHaveProperty("hookSpecificOutput.permissionDecision", "deny");
    });

    it("passes through non-Agent tool calls", () => {
      const input = {
        tool_name: "Bash",
        tool_input: { command: "echo hello" },
      } as unknown as HookInput;
      expect(processInput(input)).toEqual({});
    });

    it("includes agent suggestions in deny reason", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: null, prompt: "Debug the failing hook" },
      };
      const result = processInput(input) as any;
      const reason = result.hookSpecificOutput.permissionDecisionReason;
      expect(reason).toContain("sleuth");
      expect(reason).toContain("Always specify a specialist agent type");
    });

    it("uses description as fallback when prompt is missing", () => {
      const input: HookInput = {
        tool_name: "Agent",
        tool_input: { subagent_type: null, description: "Research codebase patterns" },
      };
      const result = processInput(input) as any;
      const reason = result.hookSpecificOutput.permissionDecisionReason;
      expect(reason).toContain("scout");
    });
  });

  describe("buildDenyResponse", () => {
    it("returns properly structured deny response", () => {
      const result = buildDenyResponse("test prompt") as any;
      expect(result.hookSpecificOutput.hookEventName).toBe("PreToolUse");
      expect(result.hookSpecificOutput.permissionDecision).toBe("deny");
      expect(typeof result.hookSpecificOutput.permissionDecisionReason).toBe("string");
    });

    it("includes full agent list in reason", () => {
      const result = buildDenyResponse("test") as any;
      const reason = result.hookSpecificOutput.permissionDecisionReason;
      expect(reason).toContain("Full agent list:");
      expect(reason).toContain("scout");
      expect(reason).toContain("maestro");
    });
  });
});
