'use client';

import { useEffect, useState, useMemo } from 'react';
import { InteractiveGraph, GRAPH_DEFS } from '@/components/visualization/InteractiveGraph';

interface ContentRendererProps {
  content: string;
  className?: string;
}

interface ContentSegment {
  type: 'markdown' | 'graph';
  content: string;
  graphType?: string;
}

// Parse content into segments
function parseContent(content: string): ContentSegment[] {
  const segments: ContentSegment[] = [];
  const graphRegex = /\{\{graph:([^}]+)\}\}/g;
  
  let lastIndex = 0;
  let match;
  
  while ((match = graphRegex.exec(content)) !== null) {
    // Add markdown before the graph
    if (match.index > lastIndex) {
      segments.push({
        type: 'markdown',
        content: content.slice(lastIndex, match.index),
      });
    }
    
    // Add the graph
    segments.push({
      type: 'graph',
      content: match[0],
      graphType: match[1].trim(),
    });
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining markdown
  if (lastIndex < content.length) {
    segments.push({
      type: 'markdown',
      content: content.slice(lastIndex),
    });
  }
  
  return segments;
}

// Render markdown segment
function MarkdownSegment({ content }: { content: string }) {
  const [html, setHtml] = useState<string>('');
  const [katex, setKatex] = useState<typeof import('katex') | null>(null);

  useEffect(() => {
    import('katex').then((k) => setKatex(k));
  }, []);

  useEffect(() => {
    if (!katex) return;

    let processed = content;

    // Convert block LaTeX ($$...$$ or \[...\])
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
      try {
        return `<div class="katex-block my-4 p-4 bg-gray-50 rounded-lg overflow-x-auto">${katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        })}</div>`;
      } catch {
        return `$$${math}$$`;
      }
    });

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
      try {
        return `<div class="katex-block my-4 p-4 bg-gray-50 rounded-lg overflow-x-auto">${katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        })}</div>`;
      } catch {
        return `\\[${math}\\]`;
      }
    });

    // Convert inline LaTeX ($...$ or \(...\))
    processed = processed.replace(/\$([^$\n]+?)\$/g, (_, math) => {
      try {
        return `<span class="katex-inline">${katex.renderToString(math.trim(), {
          displayMode: false,
          throwOnError: false,
        })}</span>`;
      } catch {
        return `$${math}$`;
      }
    });

    processed = processed.replace(/\\\(([^)]+?)\\\)/g, (_, math) => {
      try {
        return `<span class="katex-inline">${katex.renderToString(math.trim(), {
          displayMode: false,
          throwOnError: false,
        })}</span>`;
      } catch {
        return `\\(${math}\\)`;
      }
    });

    // Convert headers
    processed = processed.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-2 text-gray-900">$1</h3>');
    processed = processed.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-3 text-gray-900">$1</h2>');
    processed = processed.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-4 text-gray-900">$1</h1>');

    // Convert bold and italic
    processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold">$1</strong>');
    processed = processed.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Convert links
    processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline">$1</a>');

    // Convert code blocks
    processed = processed.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-gray-900 text-gray-100 rounded-lg p-4 my-4 overflow-x-auto"><code>$2</code></pre>');

    // Convert inline code
    processed = processed.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800">$1</code>');

    // Convert lists
    processed = processed.replace(/^- (.+)$/gm, '<li class="ml-4 text-gray-700">$1</li>');
    processed = processed.replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc my-4">$&</ul>');

    // Convert paragraphs (double newlines)
    processed = processed.split('\n\n').map(para => {
      if (para.trim() && !para.startsWith('<')) {
        return `<p class="my-4 text-gray-700 leading-relaxed">${para}</p>`;
      }
      return para;
    }).join('\n');

    setHtml(processed);
  }, [content, katex]);

  if (!html) {
    return null;
  }

  return (
    <div
      className="prose prose-slate max-w-none"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

// Render graph segment
function GraphSegment({ graphType }: { graphType: string }) {
  const graphDef = GRAPH_DEFS[graphType];
  
  if (!graphDef) {
    return (
      <div className="my-6 p-4 bg-gray-100 rounded-lg border border-gray-200">
        <p className="text-gray-500 text-sm">
          Graph type &quot;{graphType}&quot; not found. Available types: {Object.keys(GRAPH_DEFS).join(', ')}
        </p>
      </div>
    );
  }
  
  return (
    <div className="my-6">
      <InteractiveGraph
        type={graphDef.type}
        params={graphDef.params}
        title={graphDef.title}
      />
    </div>
  );
}

export function ContentRenderer({ content, className = '' }: ContentRendererProps) {
  const segments = useMemo(() => parseContent(content), [content]);
  
  return (
    <div className={className}>
      {segments.map((segment, index) => {
        if (segment.type === 'markdown') {
          return <MarkdownSegment key={index} content={segment.content} />;
        } else if (segment.type === 'graph' && segment.graphType) {
          return <GraphSegment key={index} graphType={segment.graphType} />;
        }
        return null;
      })}
    </div>
  );
}
