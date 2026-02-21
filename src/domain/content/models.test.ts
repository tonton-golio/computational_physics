import { describe, expect, it } from "vitest";
import { topicIndexFrom } from "./models";

describe("content domain models", () => {
  it("builds a topic index preserving lesson order", () => {
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

    expect(index.lessons).toEqual(["z", "a"]);
  });
});
