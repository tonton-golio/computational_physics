import { notFound } from "next/navigation";
import { getTopicSlugs, getTopicFile, TOPICS, type TopicSlug } from "@/lib/content";
import { MarkdownContent } from "@/components/content/MarkdownContent";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

// Get all topic/slug combinations for static generation
export async function generateStaticParams() {
  const params: { topic: string; slug: string }[] = [];
  
  Object.keys(TOPICS).forEach((topic) => {
    const slugs = getTopicSlugs(topic);
    slugs.forEach((slug) => {
      params.push({ topic, slug });
    });
  });
  
  return params;
}

export default function LessonPage({ params }: { params: { topic: string; slug: string } }) {
  const topicSlug = params.topic as TopicSlug;
  const topicConfig = TOPICS[topicSlug];
  
  if (!topicConfig) {
    notFound();
  }
  
  const lesson = getTopicFile(topicSlug, params.slug);
  
  if (!lesson) {
    notFound();
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-6 py-12">
        <div className="mb-6">
          <a 
            href={`/topics/${topicSlug}`} 
            className="text-sm text-primary-600 hover:underline"
          >
            ← Back to {topicConfig.title}
          </a>
        </div>
        
        <Card>
          <CardHeader>
            <div className="mb-2">
              <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${topicConfig.color}`}>
                {topicConfig.difficulty}
              </span>
            </div>
            <CardTitle className="text-2xl">{lesson.title}</CardTitle>
          </CardHeader>
          <CardContent>
            <MarkdownContent content={lesson.content} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
