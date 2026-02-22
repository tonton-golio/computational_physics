import { describe, expect, it } from "vitest";
import { resolveTopicRoute, topicHref, TOPIC_ROUTES } from "./topic-navigation";

describe("topic-navigation", () => {
  describe("TOPIC_ROUTES", () => {
    it("contains all 10 topics", () => {
      expect(TOPIC_ROUTES).toHaveLength(10);
    });

    it("has unique route slugs", () => {
      const slugs = TOPIC_ROUTES.map((r) => r.routeSlug);
      expect(new Set(slugs).size).toBe(slugs.length);
    });

    it("has unique topic IDs", () => {
      const ids = TOPIC_ROUTES.map((r) => r.topicId);
      expect(new Set(ids).size).toBe(ids.length);
    });
  });

  describe("resolveTopicRoute", () => {
    it("resolves a known topic slug", () => {
      const result = resolveTopicRoute("complex-physics");
      expect(result).toEqual({
        routeSlug: "complex-physics",
        topicId: "complex-physics",
      });
    });

    it("resolves every registered topic", () => {
      for (const route of TOPIC_ROUTES) {
        expect(resolveTopicRoute(route.routeSlug)).toEqual(route);
      }
    });

    it("returns null for unknown slug", () => {
      expect(resolveTopicRoute("nonexistent-topic")).toBeNull();
    });

    it("returns null for empty string", () => {
      expect(resolveTopicRoute("")).toBeNull();
    });

    it("is case-sensitive", () => {
      expect(resolveTopicRoute("Complex-Physics")).toBeNull();
    });
  });

  describe("topicHref", () => {
    it("builds topic-only path", () => {
      expect(topicHref("complex-physics")).toBe("/topics/complex-physics");
    });

    it("builds topic + lesson path", () => {
      expect(topicHref("complex-physics", "percolation")).toBe(
        "/topics/complex-physics/percolation"
      );
    });

    it("handles undefined slug same as omitted", () => {
      expect(topicHref("quantum-optics", undefined)).toBe(
        "/topics/quantum-optics"
      );
    });

    it("handles empty string slug as truthy (returns lesson path)", () => {
      // empty string is falsy, so it falls through to topic-only
      expect(topicHref("quantum-optics", "")).toBe("/topics/quantum-optics");
    });
  });
});
