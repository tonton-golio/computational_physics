import { getTopicFile } from '@/lib/content';
import { MarkdownContent } from '@/components/content/MarkdownContent';

export default function LearnToCodePage() {
  const content = getTopicFile('learn-to-code', 'initial');

  if (!content) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8">Learn to Code</h1>
        <p>Content not found.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-8">{content.title}</h1>
      <MarkdownContent content={content.content} className="max-w-4xl mx-auto" />
    </div>
  );
}