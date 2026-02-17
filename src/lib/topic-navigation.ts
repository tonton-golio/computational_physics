import { TOPICS } from "@/lib/content";

export interface TopicRouteConfig {
  routeSlug: string;
  contentId: keyof typeof TOPICS;
}

export const TOPIC_ROUTES: TopicRouteConfig[] = [
  { routeSlug: "applied-statistics", contentId: "applied-statistics" },
  { routeSlug: "applied-machine-learning", contentId: "applied-machine-learning" },
  { routeSlug: "dynamical-models", contentId: "dynamical-models" },
  { routeSlug: "scientific-computing", contentId: "scientific-computing" },
  { routeSlug: "complex-physics", contentId: "complex-physics" },
  { routeSlug: "continuum-mechanics", contentId: "continuum-mechanics" },
  { routeSlug: "inverse-problems", contentId: "inverse-problems" },
  { routeSlug: "quantum-optics", contentId: "quantum-optics" },
  { routeSlug: "online-reinforcement", contentId: "online-reinforcement-learning" },
  { routeSlug: "advanced-deep-learning", contentId: "advanced-deep-learning" },
];

export function resolveTopicRoute(routeSlug: string): TopicRouteConfig | null {
  return TOPIC_ROUTES.find((topic) => topic.routeSlug === routeSlug) ?? null;
}

export function topicHref(routeSlug: string, slug?: string): string {
  if (slug) return `/topics/${routeSlug}/${slug}`;
  return `/topics/${routeSlug}`;
}
