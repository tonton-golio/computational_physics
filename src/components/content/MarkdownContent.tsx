"use client";

import { useEffect, useState } from "react";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

// Simple markdown to HTML converter with LaTeX support
export function MarkdownContent({ content, className = "" }: MarkdownContentProps) {
  const [html, setHtml] = useState<string>("");
  const [katex, setKatex] = useState<typeof import("katex") | null>(null);

  useEffect(() => {
    import("katex").then((k) => setKatex(k));
  }, []);

  useEffect(() => {
    if (!katex) return;

    let processed = content;

    // Convert block LaTeX ($$...$$ or \[...\])
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
      try {
        return `<div class="katex-block">${katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        })}</div>`;
      } catch {
        return `$$${math}$$`;
      }
    });

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
      try {
        return `<div class="katex-block">${katex.renderToString(math.trim(), {
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
    processed = processed.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-2">$1</h3>');
    processed = processed.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-3">$1</h2>');
    processed = processed.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-4">$1</h1>');

    // Convert bold and italic
    processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Convert links
    processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-primary-600 hover:underline">$1</a>');

    // Convert code blocks
    processed = processed.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-gray-900 text-gray-100 rounded-lg p-4 my-4 overflow-x-auto"><code>$2</code></pre>');

    // Convert inline code
    processed = processed.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>');

    // Convert lists
    processed = processed.replace(/^- (.+)$/gm, '<li class="ml-4">$1</li>');
    processed = processed.replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc my-4">$&</ul>');

    // Convert paragraphs (double newlines)
    processed = processed.split('\n\n').map(para => {
      if (para.trim() && !para.startsWith('<')) {
        return `<p class="my-4">${para}</p>`;
      }
      return para;
    }).join('\n');

    setHtml(processed);
  }, [content, katex]);

  if (!html) {
    return <div className={className}>Loading...</div>;
  }

  return (
    <div 
      className={`prose prose-slate max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
