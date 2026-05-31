import type { MetadataRoute } from "next";
import { getTopicIndexes } from "@/features/content/content-gateway";
import { topicHref } from "@/lib/topic-navigation";

const BASE_URL = "https://koalabrain.org";

export const revalidate = 3600;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const indexes = await getTopicIndexes();

  const entries: MetadataRoute.Sitemap = [
    { url: `${BASE_URL}/`, changeFrequency: "weekly", priority: 1 },
    { url: `${BASE_URL}/topics`, changeFrequency: "weekly", priority: 0.9 },
  ];

  for (const { topic, lessons } of indexes) {
    entries.push({
      url: `${BASE_URL}${topicHref(topic.id)}`,
      changeFrequency: "weekly",
      priority: 0.7,
    });
    for (const slug of lessons) {
      entries.push({
        url: `${BASE_URL}${topicHref(topic.id, slug)}`,
        changeFrequency: "monthly",
        priority: 0.6,
      });
    }
  }

  return entries;
}
