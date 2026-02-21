import { describe, expect, it } from "vitest";
import { topicIndexFrom } from "./models";

const makeTopic = (overrides = {}) => ({
  id: "topic",
  title: "Topic",
  description: "desc",
  difficulty: "Beginner" as const,
  color: "blue",
  relatedTopics: [],
  ...overrides,
});

describe("content domain models", () => {
  it("builds a topic index preserving lesson order", () => {
    const index = topicIndexFrom(makeTopic(), ["z", "a"]);
    expect(index.lessons).toEqual(["z", "a"]);
  });

  it("copies lessons array (not a reference)", () => {
    const original = ["x", "y"];
    const index = topicIndexFrom(makeTopic(), original);
    original.push("z");
    expect(index.lessons).toEqual(["x", "y"]);
  });

  it("handles empty lessons array", () => {
    const index = topicIndexFrom(makeTopic(), []);
    expect(index.lessons).toEqual([]);
  });

  it("attaches topic definition to index", () => {
    const topic = makeTopic({ id: "my-topic", title: "My Topic" });
    const index = topicIndexFrom(topic, ["lesson-1"]);
    expect(index.topic.id).toBe("my-topic");
    expect(index.topic.title).toBe("My Topic");
  });

  it("preserves all topic definition fields", () => {
    const topic = makeTopic({
      difficulty: "Expert",
      color: "red",
      relatedTopics: ["a", "b"],
    });
    const index = topicIndexFrom(topic, []);
    expect(index.topic.difficulty).toBe("Expert");
    expect(index.topic.color).toBe("red");
    expect(index.topic.relatedTopics).toEqual(["a", "b"]);
  });
});
