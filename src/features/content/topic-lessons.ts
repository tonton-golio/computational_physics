import { listLessonSlugs, readLessonDocument } from "@/infra/content/file-content-repository";
import {
  TOPIC_LESSON_ORDER,
  LANDING_PAGE_PRIORITY,
  getLessonSummary,
  isLandingPage,
  type SearchableLesson,
} from "@/lib/topic-navigation.server";

// Re-export so consumers don't need to import from topic-navigation.server directly
export { getLessonSummary, isLandingPage };

export function getOrderedLessonSlugs(topicId: string): string[] {
  const slugs = listLessonSlugs(topicId);
  const topicOrder = TOPIC_LESSON_ORDER[topicId];

  if (topicOrder) {
    const orderIndex = new Map(topicOrder.map((slug, i) => [slug, i + 1]));
    return slugs.sort((a, b) => {
      const pa = LANDING_PAGE_PRIORITY.get(a) ?? orderIndex.get(a) ?? 1000;
      const pb = LANDING_PAGE_PRIORITY.get(b) ?? orderIndex.get(b) ?? 1000;
      if (pa !== pb) return pa - pb;
      return a.localeCompare(b);
    });
  }

  return slugs.sort((a, b) => {
    const pa = LANDING_PAGE_PRIORITY.get(a) ?? 0;
    const pb = LANDING_PAGE_PRIORITY.get(b) ?? 0;
    if (pa !== pb) return pa - pb;
    return a.localeCompare(b);
  });
}

export function getLandingPageSlug(topicId: string): string | null {
  const slugs = listLessonSlugs(topicId);
  let best: string | null = null;
  let bestPriority = Infinity;
  for (const slug of slugs) {
    const priority = LANDING_PAGE_PRIORITY.get(slug);
    if (priority !== undefined && priority < bestPriority) {
      best = slug;
      bestPriority = priority;
    }
  }
  return best;
}

export function getLessonsForTopic(topicId: string): Array<{ slug: string; title: string; summary?: string }> {
  return getOrderedLessonSlugs(topicId)
    .map((slug) => {
      const doc = readLessonDocument(topicId, slug);
      if (!doc) return null;
      const summary = getLessonSummary(topicId, slug);
      return { slug, title: doc.title, ...(summary ? { summary } : {}) };
    })
    .filter((lesson): lesson is { slug: string; title: string; summary?: string } => lesson !== null);
}

export function getSearchableLessonsForTopic(topicId: string): SearchableLesson[] {
  return getOrderedLessonSlugs(topicId)
    .filter((slug) => !isLandingPage(slug))
    .map((slug) => {
      const doc = readLessonDocument(topicId, slug);
      if (!doc) return null;

      const summary = getLessonSummary(topicId, slug);
      return {
        slug,
        title: doc.title,
        ...(summary ? { summary } : {}),
        searchableText: `${doc.title}\n${doc.content}`.toLowerCase(),
      };
    })
    .filter((lesson): lesson is SearchableLesson => lesson !== null);
}
