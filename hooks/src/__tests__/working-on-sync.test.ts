import { describe, expect, it } from "vitest";
import {
  parseCreatedTaskId,
  pickTodoInProgress,
  deriveWorkingOn,
} from "../working-on-sync.ts";

const EMPTY = { tasks: {} as Record<string, string>, currentId: null as string | null };

describe("parseCreatedTaskId", () => {
  it("extracts the id from a created-task response string", () => {
    expect(parseCreatedTaskId("Task #1 created successfully: Backport")).toBe("1");
    expect(parseCreatedTaskId("Task #42 created")).toBe("42");
  });
  it("extracts from an object response", () => {
    expect(parseCreatedTaskId({ text: "Task #7 created successfully" })).toBe("7");
  });
  it("ignores lowercase 'task #N' (TaskUpdate response) and non-matches", () => {
    // TaskUpdate responds "Updated task #1 status" — must NOT be read as a created id.
    expect(parseCreatedTaskId("Updated task #1 status")).toBeNull();
    expect(parseCreatedTaskId("nothing here")).toBeNull();
    expect(parseCreatedTaskId(undefined)).toBeNull();
  });
});

describe("pickTodoInProgress", () => {
  it("returns activeForm of the in-progress todo", () => {
    expect(
      pickTodoInProgress([
        { content: "A", status: "completed" },
        { content: "Do B", activeForm: "Doing B", status: "in_progress" },
      ]),
    ).toBe("Doing B");
  });
  it("falls back to content when activeForm is absent", () => {
    expect(pickTodoInProgress([{ content: "Do B", status: "in_progress" }])).toBe("Do B");
  });
  it("returns '' when none is in progress", () => {
    expect(pickTodoInProgress([{ content: "A", status: "pending" }])).toBe("");
    expect(pickTodoInProgress([])).toBe("");
    expect(pickTodoInProgress(undefined)).toBe("");
  });
});

describe("deriveWorkingOn — TodoWrite", () => {
  it("sets working_on to the in-progress todo's activeForm", () => {
    const r = deriveWorkingOn(
      {
        tool_name: "TodoWrite",
        tool_input: { todos: [{ content: "X", activeForm: "Doing X", status: "in_progress" }] },
      },
      EMPTY,
    );
    expect(r.workingOn).toBe("Doing X");
  });
  it("clears working_on ('') when no todo is in progress", () => {
    const r = deriveWorkingOn(
      { tool_name: "TodoWrite", tool_input: { todos: [{ content: "X", status: "completed" }] } },
      EMPTY,
    );
    expect(r.workingOn).toBe("");
  });
});

describe("deriveWorkingOn — native Task tools", () => {
  it("TaskCreate records the label in cache but does NOT write (pending)", () => {
    const r = deriveWorkingOn(
      {
        tool_name: "TaskCreate",
        tool_input: { subject: "Fix 65", activeForm: "Fixing 65" },
        tool_response: "Task #2 created successfully: Fix 65",
      },
      EMPTY,
    );
    expect(r.workingOn).toBeNull();
    expect(r.cache.tasks["2"]).toBe("Fixing 65");
  });

  it("TaskUpdate -> in_progress resolves the cached label", () => {
    const cache = { tasks: { "1": "Backporting", "2": "Fixing 65" }, currentId: null };
    const r = deriveWorkingOn(
      { tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "in_progress" } },
      cache,
    );
    expect(r.workingOn).toBe("Backporting");
    expect(r.cache.currentId).toBe("1");
  });

  it("picks the RIGHT task when several were created up front", () => {
    // create #1 then #2, then mark #1 in_progress: working_on must be #1's label, not #2's
    let cache = EMPTY;
    cache = deriveWorkingOn(
      { tool_name: "TaskCreate", tool_input: { activeForm: "Backporting" }, tool_response: "Task #1 created" },
      cache,
    ).cache;
    cache = deriveWorkingOn(
      { tool_name: "TaskCreate", tool_input: { activeForm: "Fixing 65" }, tool_response: "Task #2 created" },
      cache,
    ).cache;
    const r = deriveWorkingOn(
      { tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "in_progress" } },
      cache,
    );
    expect(r.workingOn).toBe("Backporting");
  });

  it("TaskUpdate -> in_progress with unknown id does not write", () => {
    const r = deriveWorkingOn(
      { tool_name: "TaskUpdate", tool_input: { taskId: "99", status: "in_progress" } },
      EMPTY,
    );
    expect(r.workingOn).toBeNull();
  });

  it("completing the current task clears working_on", () => {
    const cache = { tasks: { "1": "Backporting" }, currentId: "1" };
    const r = deriveWorkingOn(
      { tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "completed" } },
      cache,
    );
    expect(r.workingOn).toBe("");
    expect(r.cache.currentId).toBeNull();
  });

  it("completing a NON-current task leaves working_on untouched", () => {
    const cache = { tasks: { "1": "Backporting", "2": "Fixing 65" }, currentId: "2" };
    const r = deriveWorkingOn(
      { tool_name: "TaskUpdate", tool_input: { taskId: "1", status: "completed" } },
      cache,
    );
    expect(r.workingOn).toBeNull();
    expect(r.cache.currentId).toBe("2");
  });

  it("does not mutate the input cache (immutability)", () => {
    const cache = { tasks: { "1": "A" }, currentId: null };
    deriveWorkingOn(
      { tool_name: "TaskCreate", tool_input: { activeForm: "B" }, tool_response: "Task #2 created" },
      cache,
    );
    expect(cache.tasks).toEqual({ "1": "A" });
  });
});
