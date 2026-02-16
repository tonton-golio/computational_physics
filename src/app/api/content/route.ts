import { NextResponse } from 'next/server';
import { getTopicFile, getTopicSlugs, TOPICS, TopicSlug } from '@/lib/content';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const topic = searchParams.get('topic');
  const slug = searchParams.get('slug');
  
  // Get all topics overview
  if (!topic) {
    const topicsOverview = Object.entries(TOPICS).map(([key, value]) => ({
      id: key,
      ...value,
      lessons: getTopicSlugs(key),
    }));
    return NextResponse.json({ topics: topicsOverview });
  }
  
  // Get specific content
  if (topic && slug) {
    const content = getTopicFile(topic, slug);
    if (!content) {
      return NextResponse.json({ error: 'Content not found' }, { status: 404 });
    }
    return NextResponse.json({ content });
  }
  
  // Get all content for a topic
  if (topic) {
    if (!(topic in TOPICS)) {
      return NextResponse.json({ error: 'Topic not found' }, { status: 404 });
    }
    const slugs = getTopicSlugs(topic);
    const files = slugs.map(s => getTopicFile(topic, s)).filter(Boolean);
    return NextResponse.json({ 
      topic: TOPICS[topic as TopicSlug],
      lessons: files 
    });
  }
  
  return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
}
