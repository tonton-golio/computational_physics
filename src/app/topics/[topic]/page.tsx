import { notFound, redirect } from "next/navigation";
import { resolveTopicRoute, topicHref, TOPIC_ROUTES } from "@/lib/topic-navigation";
import { getOrderedLessonSlugs } from "@/lib/topic-navigation.server";

interface TopicPageProps {
  params: Promise<{ topic: string }>;
}

export function generateStaticParams() {
  return TOPIC_ROUTES.map((topic) => ({ topic: topic.routeSlug }));
}

export default async function TopicPage({ params }: TopicPageProps) {
  const { topic } = await params;
  const route = resolveTopicRoute(topic);
  if (!route) notFound();

  const slugs = getOrderedLessonSlugs(route.contentId);
  if (!slugs.length) {
    redirect("/topics");
  }

  redirect(topicHref(route.routeSlug, slugs[0]));
}
