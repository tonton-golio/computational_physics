import { describe, expect, it } from "vitest";
import { listLessonSlugs, readLessonDocument } from "./file-content-repository";

describe("file-content-repository", () => {
  describe("listLessonSlugs", () => {
    it("returns slugs for a known topic", () => {
      const slugs = listLessonSlugs("complex-physics");
      expect(slugs.length).toBeGreaterThan(0);
      expect(slugs.every((s) => typeof s === "string")).toBe(true);
    });

    it("strips file extensions from slugs", () => {
      const slugs = listLessonSlugs("complex-physics");
      for (const slug of slugs) {
        expect(slug).not.toMatch(/\.(md|txt)$/);
      }
    });

    it("returns empty array for nonexistent topic", () => {
      expect(listLessonSlugs("nonexistent-topic-xyz")).toEqual([]);
    });

    it("returns slugs for all 10 topics", () => {
      const topics = [
        "complex-physics",
        "advanced-deep-learning",
        "applied-machine-learning",
        "applied-statistics",
        "continuum-mechanics",
        "dynamical-models",
        "inverse-problems",
        "online-reinforcement-learning",
        "quantum-optics",
        "scientific-computing",
      ];
      for (const topicId of topics) {
        const slugs = listLessonSlugs(topicId);
        expect(slugs.length, `${topicId} should have lessons`).toBeGreaterThan(0);
      }
    });
  });

  describe("readLessonDocument", () => {
    it("reads a known lesson document", () => {
      const doc = readLessonDocument("complex-physics", "percolation");
      expect(doc).not.toBeNull();
      expect(doc!.topicId).toBe("complex-physics");
      expect(doc!.slug).toBe("percolation");
      expect(doc!.title).toBeTruthy();
      expect(doc!.content).toBeTruthy();
    });

    it("returns null for nonexistent slug", () => {
      expect(readLessonDocument("complex-physics", "nonexistent-slug")).toBeNull();
    });

    it("returns null for nonexistent topic", () => {
      expect(readLessonDocument("nonexistent-topic", "anything")).toBeNull();
    });

    it("extracts title from markdown heading", () => {
      const doc = readLessonDocument("complex-physics", "percolation");
      if (doc) {
        // Title should be non-empty — either from frontmatter or # heading
        expect(doc.title.length).toBeGreaterThan(0);
      }
    });

    it("includes meta as a record", () => {
      const doc = readLessonDocument("complex-physics", "percolation");
      if (doc) {
        expect(typeof doc.meta).toBe("object");
      }
    });

    it("content field contains markdown body", () => {
      const doc = readLessonDocument("complex-physics", "percolation");
      if (doc) {
        expect(doc.content.length).toBeGreaterThan(100);
      }
    });
  });
});
