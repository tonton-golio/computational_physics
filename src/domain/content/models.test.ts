import { describe, expect, it } from "vitest";
import { sortLessonsBySlug, topicIndexFrom } from "./models";

describe("content domain models", () => {
  it("sorts lessons deterministically by slug", () => {
    const sorted = sortLessonsBySlug([
      { topicId: "x", slug: "b", title: "B", content: "", meta: {} },
      { topicId: "x", slug: "a", title: "A", content: "", meta: {} },
    ]);
    expect(sorted.map((entry) => entry.slug)).toEqual(["a", "b"]);
  });

  it("builds a topic index with sorted lessons", () => {
    const index = topicIndexFrom(
      {
        id: "topic",
        title: "Topic",
        description: "desc",
        difficulty: "Beginner",
        color: "blue",
        relatedTopics: [],
      },
      ["z", "a"]
    );

    expect(index.lessons).toEqual(["a", "z"]);
  });
});
