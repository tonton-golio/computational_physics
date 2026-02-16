"use client";

import { useEffect, useState, Suspense } from "react";
import { InteractiveGraph, GRAPH_DEFS } from "../visualization/InteractiveGraph";
import { EIGENVALUE_SIMULATIONS } from "../visualization/EigenvalueSimulations";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

// Loading placeholder for simulations
function SimulationLoading() {
  return (
    <div className="w-full h-80 bg-[#151525] rounded-lg flex items-center justify-center">
      <div className="text-gray-400">Loading simulation...</div>
    </div>
  );
}

// Error placeholder
function SimulationError({ id, type }: { id: string; type: string }) {
  return (
    <div className="w-full h-64 bg-[#151525] rounded-lg flex flex-col items-center justify-center text-gray-400">
      <span className="text-red-400 mb-2">Unknown placeholder type</span>
      <span className="text-sm">[[{type} {id}]]</span>
    </div>
  );
}

// Render simulation based on type and id
function renderPlaceholder(type: string, id: string): React.ReactNode {
  if (type === "simulation") {
    // Check if it's an eigenvalue simulation
    const EigenSim = EIGENVALUE_SIMULATIONS[id];
    if (EigenSim) {
      return (
        <Suspense fallback={<SimulationLoading />}>
          <EigenSim id={id} />
        </Suspense>
      );
    }
    
    // Check if it's a general graph type
    const graphDef = GRAPH_DEFS[id];
    if (graphDef) {
      return <InteractiveGraph type={id} params={graphDef.params} />;
    }
    
    // Default: try to render as interactive graph
    return <InteractiveGraph type={id} />;
  }
  
  if (type === "figure") {
    // Figures are static images or diagrams
    // For now, render a placeholder with the figure ID
    return (
      <div className="w-full bg-[#151525] rounded-lg p-8 text-center">
        <div className="text-gray-400 mb-2">📊 Figure</div>
        <div className="text-sm text-gray-500">{id}</div>
        <div className="text-xs text-gray-600 mt-4">
          Add image to <code className="bg-[#0a0a15] px-2 py-1 rounded">public/figures/{id}.png</code>
        </div>
      </div>
    );
  }
  
  return <SimulationError id={id} type={type} />;
}

// Simple markdown to HTML converter with LaTeX and placeholder support
export function MarkdownContent({ content, className = "" }: MarkdownContentProps) {
  const [processedContent, setProcessedContent] = useState<{
    html: string;
    placeholders: Array<{ type: string; id: string; index: number }>;
  }>({ html: "", placeholders: [] });
  const [katex, setKatex] = useState<typeof import("katex") | null>(null);

  useEffect(() => {
    import("katex").then((k) => setKatex(k));
  }, []);

  useEffect(() => {
    if (!katex) return;

    let processed = content;
    const placeholders: Array<{ type: string; id: string; index: number }> = [];

    // Extract and replace placeholders with temporary markers
    let placeholderIndex = 0;

    processed = processed.replace(/\[\[(simulation|figure)\s+([^\]]+)\]\]/g, (match, type, id) => {
      const marker = `__PLACEHOLDER_${placeholderIndex}__`;
      placeholders.push({ type, id: id.trim(), index: placeholderIndex });
      placeholderIndex++;
      return marker;
    });

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
    processed = processed.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-2 text-white">$1</h3>');
    processed = processed.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-3 text-white">$1</h2>');
    processed = processed.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-4 text-white">$1</h1>');

    // Convert bold and italic
    processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Convert links
    processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>');

    // Convert code blocks
    processed = processed.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-[#0a0a15] text-gray-100 rounded-lg p-4 my-4 overflow-x-auto text-sm"><code>$2</code></pre>');

    // Convert inline code
    processed = processed.replace(/`([^`]+)`/g, '<code class="bg-[#0a0a15] px-1.5 py-0.5 rounded text-sm font-mono text-blue-300">$1</code>');

    // Convert lists
    processed = processed.replace(/^- (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');
    processed = processed.replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc my-4 space-y-1">$&</ul>');

    // Convert numbered lists
    processed = processed.replace(/^\d+\. (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');
    
    // Convert tables
    processed = processed.replace(/^\|(.+)\|$/gm, (match, content) => {
      const cells = content.split('|').map((c: string) => c.trim());
      const isHeader = cells.some((c: string) => c.match(/^-+$/));
      if (isHeader) return '';
      const tag = 'td';
      return `<tr>${cells.map((c: string) => `<${tag} class="border border-gray-700 px-3 py-2 text-gray-300">${c}</${tag}>`).join('')}</tr>`;
    });
    
    // Wrap tables
    processed = processed.replace(/(<tr>.*<\/tr>\n?)+/g, '<table class="w-full border-collapse my-4 bg-[#0a0a15] rounded">$&</table>');
    
    // Add table headers (first row after <table>)
    processed = processed.replace(/<table([^>]*)>(<tr>.*?<\/tr>)/g, (match, attrs, firstRow) => {
      const headerRow = firstRow.replace(/<td/g, '<th').replace(/<\/td>/g, '</th>');
      return `<table${attrs}>${headerRow}`;
    });

    // Convert horizontal rules
    processed = processed.replace(/^---$/gm, '<hr class="border-gray-700 my-8" />');

    // Convert paragraphs (double newlines)
    processed = processed.split('\n\n').map(para => {
      if (para.trim() && !para.startsWith('<')) {
        return `<p class="my-4 text-gray-300 leading-relaxed">${para}</p>`;
      }
      return para;
    }).join('\n');

    setProcessedContent({ html: processed, placeholders });
  }, [content, katex]);

  if (!processedContent.html) {
    return <div className={className}>Loading...</div>;
  }

  // Split content by placeholder markers and render with components
  const parts = processedContent.html.split(/__PLACEHOLDER_(\d+)__/);
  
  return (
    <div className={`prose prose-invert max-w-none ${className}`}>
      {parts.map((part, index) => {
        if (index % 2 === 0) {
          // Regular HTML content
          return (
            <div
              key={index}
              dangerouslySetInnerHTML={{ __html: part }}
            />
          );
        } else {
          // Placeholder component
          const placeholderIndex = parseInt(part);
          const placeholder = processedContent.placeholders[placeholderIndex];
          if (placeholder) {
            return (
              <div key={index} className="my-8">
                {renderPlaceholder(placeholder.type, placeholder.id)}
              </div>
            );
          }
          return null;
        }
      })}
    </div>
  );
}
