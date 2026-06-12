// src/edit-context-inject.ts
import { readFileSync, existsSync } from "fs";
import { basename } from "path";
var SYMBOL_INDEX_FILE = "/tmp/claude-symbol-index/symbols.json";
var symbolIndex = null;
function loadSymbolIndex() {
  if (symbolIndex !== null) return symbolIndex;
  try {
    if (existsSync(SYMBOL_INDEX_FILE)) {
      symbolIndex = JSON.parse(readFileSync(SYMBOL_INDEX_FILE, "utf-8"));
      return symbolIndex;
    }
  } catch {
  }
  symbolIndex = {};
  return symbolIndex;
}
function getFileSymbols(filePath) {
  const index = loadSymbolIndex();
  const result = { functions: [], classes: [], variables: [] };
  for (const [name, entry] of Object.entries(index)) {
    if (entry.location.includes(filePath) || entry.location.includes(basename(filePath))) {
      if (entry.type === "function") result.functions.push(name);
      else if (entry.type === "class") result.classes.push(name);
      else if (entry.type === "variable") result.variables.push(name);
    }
  }
  return result;
}
async function main() {
  const input = JSON.parse(readFileSync(0, "utf-8"));
  if (input.tool_name !== "Edit") {
    console.log("{}");
    return;
  }
  const filePath = input.tool_input.file_path;
  if (!filePath) {
    console.log("{}");
    return;
  }
  const symbols = getFileSymbols(filePath);
  const total = symbols.functions.length + symbols.classes.length + symbols.variables.length;
  if (total === 0) {
    console.log("{}");
    return;
  }
  const parts = [];
  if (symbols.classes.length > 0) {
    parts.push(`Classes: ${symbols.classes.slice(0, 10).join(", ")}${symbols.classes.length > 10 ? "..." : ""}`);
  }
  if (symbols.functions.length > 0) {
    parts.push(`Functions: ${symbols.functions.slice(0, 15).join(", ")}${symbols.functions.length > 15 ? "..." : ""}`);
  }
  if (symbols.variables.length > 0 && symbols.variables.length <= 10) {
    parts.push(`Variables: ${symbols.variables.join(", ")}`);
  }
  const output = {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      additionalContext: `[Edit context: ${basename(filePath)} has ${total} symbols]
${parts.join("\n")}`
    }
  };
  console.log(JSON.stringify(output));
}
main().catch(() => console.log("{}"));
