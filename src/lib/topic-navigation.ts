import { TOPICS } from "@/lib/topic-config";

export interface TopicRouteConfig {
  routeSlug: string;
  topicId: keyof typeof TOPICS;
}

export const TOPIC_ROUTES: TopicRouteConfig[] = Object.keys(TOPICS).map(
  (id) => ({ routeSlug: id, topicId: id as keyof typeof TOPICS }),
);

export function resolveTopicRoute(routeSlug: string): TopicRouteConfig | null {
  return TOPIC_ROUTES.find((topic) => topic.routeSlug === routeSlug) ?? null;
}

export function topicHref(routeSlug: string, slug?: string): string {
  if (slug) return `/topics/${routeSlug}/${slug}`;
  return `/topics/${routeSlug}`;
}
