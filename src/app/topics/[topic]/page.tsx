import { notFound } from "next/navigation";
import Link from "next/link";
import { getTopicSlugs, getTopicFile, TOPICS, type TopicSlug } from "@/lib/content";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

// Get all topic slugs for static generation
export async function generateStaticParams() {
  return Object.keys(TOPICS).map((topic) => ({ topic }));
}

export default function TopicDetailPage({ params }: { params: { topic: string } }) {
  const topicSlug = params.topic as TopicSlug;
  const topicConfig = TOPICS[topicSlug];
  
  if (!topicConfig) {
    notFound();
  }
  
  const lessons = getTopicSlugs(topicSlug);
  
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 py-12">
        {/* Header */}
        <div className="mb-8">
          <div className="mb-4">
            <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${topicConfig.color}`}>
              {topicConfig.difficulty} Level
            </span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900">{topicConfig.title}</h1>
          <p className="mt-4 text-lg text-gray-600">{topicConfig.description}</p>
        </div>

        {/* Lessons List */}
        <div>
          <h2 className="mb-4 text-2xl font-semibold text-gray-900">Content ({lessons.length} items)</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {lessons.map((slug, index) => {
              const content = getTopicFile(topicSlug, slug);
              return (
                <Link key={slug} href={`/topics/${topicSlug}/${slug}`}>
                  <Card className="h-full transition-shadow hover:shadow-md cursor-pointer">
                    <CardHeader>
                      <div className="flex items-start gap-3">
                        <span className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium ${topicConfig.color}`}>
                          {index + 1}
                        </span>
                        <CardTitle className="text-base">
                          {content?.title || slug}
                        </CardTitle>
                      </div>
                    </CardHeader>
                  </Card>
                </Link>
              );
            })}
          </div>
          
          {lessons.length === 0 && (
            <Card className="text-center py-12">
              <CardContent>
                <p className="text-gray-500">Content coming soon...</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
