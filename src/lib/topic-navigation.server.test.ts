import { describe, expect, it } from "vitest";
import {
  getLessonSummary,
  isLandingPage,
  LESSON_SUMMARIES,
} from "./topic-navigation.server";
import {
  getOrderedLessonSlugs,
  getLandingPageSlug,
  getLessonsForTopic,
} from "@/features/content/topic-lessons";

describe("topic-navigation.server", () => {
  describe("getLessonSummary", () => {
    it("returns summary for known topic and slug", () => {
      const summary = getLessonSummary("complex-physics", "percolation");
      expect(summary).toBeTruthy();
      expect(typeof summary).toBe("string");
    });

    it("returns undefined for unknown topic", () => {
      expect(getLessonSummary("nonexistent", "anything")).toBeUndefined();
    });

    it("returns undefined for unknown slug in known topic", () => {
      expect(getLessonSummary("complex-physics", "nonexistent-slug")).toBeUndefined();
    });
  });

  describe("LESSON_SUMMARIES", () => {
    it("has summaries for all 10 topics", () => {
      const topicIds = Object.keys(LESSON_SUMMARIES);
      expect(topicIds.length).toBe(10);
    });

    it("each summary is a non-empty string", () => {
      for (const [topicId, lessons] of Object.entries(LESSON_SUMMARIES)) {
        for (const [slug, summary] of Object.entries(lessons)) {
          expect(summary, `${topicId}/${slug}`).toBeTruthy();
          expect(typeof summary, `${topicId}/${slug}`).toBe("string");
        }
      }
    });
  });

  describe("getOrderedLessonSlugs", () => {
    it("returns ordered slugs for a known topic", () => {
      const slugs = getOrderedLessonSlugs("complex-physics");
      expect(slugs.length).toBeGreaterThan(0);
      expect(Array.isArray(slugs)).toBe(true);
    });

    it("returns slugs for all 10 topics", () => {
      const topicIds = [
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
      for (const topicId of topicIds) {
        const slugs = getOrderedLessonSlugs(topicId);
        expect(slugs.length, `${topicId} should have lessons`).toBeGreaterThan(0);
      }
    });

    it("returns empty array for nonexistent topic", () => {
      expect(getOrderedLessonSlugs("nonexistent")).toEqual([]);
    });

    it("respects explicit lesson ordering", () => {
      const slugs = getOrderedLessonSlugs("complex-physics");
      const statisticalMechanicsIdx = slugs.indexOf("statisticalMechanics");
      const percolationIdx = slugs.indexOf("percolation");
      // statisticalMechanics comes before percolation in TOPIC_LESSON_ORDER
      if (statisticalMechanicsIdx !== -1 && percolationIdx !== -1) {
        expect(statisticalMechanicsIdx).toBeLessThan(percolationIdx);
      }
    });
  });

  describe("getLandingPageSlug", () => {
    it("returns 'home' for topics with a home.md file", () => {
      expect(getLandingPageSlug("complex-physics")).toBe("home");
      expect(getLandingPageSlug("advanced-deep-learning")).toBe("home");
      expect(getLandingPageSlug("scientific-computing")).toBe("home");
    });

    it("returns null for nonexistent topic", () => {
      expect(getLandingPageSlug("nonexistent")).toBeNull();
    });
  });

  describe("isLandingPage", () => {
    it("identifies home as landing page", () => {
      expect(isLandingPage("home")).toBe(true);
    });

    it("identifies intro as landing page", () => {
      expect(isLandingPage("intro")).toBe(true);
    });

    it("identifies introduction as landing page", () => {
      expect(isLandingPage("introduction")).toBe(true);
    });

    it("identifies landingPage as landing page", () => {
      expect(isLandingPage("landingPage")).toBe(true);
    });

    it("rejects regular lesson slugs", () => {
      expect(isLandingPage("percolation")).toBe(false);
      expect(isLandingPage("fractals")).toBe(false);
    });
  });

  describe("getLessonsForTopic", () => {
    it("returns lessons with title and slug", () => {
      const lessons = getLessonsForTopic("complex-physics");
      expect(lessons.length).toBeGreaterThan(0);
      for (const lesson of lessons) {
        expect(lesson.slug).toBeTruthy();
        expect(lesson.title).toBeTruthy();
      }
    });

    it("includes summaries when available", () => {
      const lessons = getLessonsForTopic("complex-physics");
      const withSummary = lessons.filter((l) => l.summary);
      expect(withSummary.length).toBeGreaterThan(0);
    });

    it("returns empty array for nonexistent topic", () => {
      expect(getLessonsForTopic("nonexistent")).toEqual([]);
    });
  });
});
