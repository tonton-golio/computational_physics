import Link from "next/link";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { TOPICS } from "@/lib/content";

export default function TopicsPage() {
  const topics = Object.entries(TOPICS).map(([slug, config]) => ({
    slug,
    ...config,
  }));

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <h1 className="text-3xl font-bold text-gray-900">Topics</h1>
        <p className="mt-2 text-gray-600">
          Explore our comprehensive collection of computational physics topics
        </p>

        <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {topics.map((topic) => (
            <Link key={topic.slug} href={`/topics/${topic.slug}`}>
              <Card className="h-full transition-all hover:shadow-lg hover:border-primary-300 cursor-pointer">
                <CardHeader>
                  <div className="mb-2 flex items-center justify-between">
                    <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${topic.color}`}>
                      {topic.difficulty}
                    </span>
                  </div>
                  <CardTitle>{topic.title}</CardTitle>
                  <CardDescription>{topic.description}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
