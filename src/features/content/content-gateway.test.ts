import { describe, expect, it } from "vitest";
import { getTopicIndexes, getTopicLessons } from "./content-gateway";

describe("content gateway", () => {
  it("loads topic index", async () => {
    const topics = await getTopicIndexes();
    expect(topics.length).toBeGreaterThan(0);
    expect(topics[0]?.topic.id).toBeTruthy();
  });

  it("loads lessons for a known topic", async () => {
    const result = await getTopicLessons("complex-physics");
    expect(result.topic.id).toBe("complex-physics");
    expect(Array.isArray(result.lessons)).toBe(true);
  });
});
