"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { CodeEditor } from "../ui/code-editor";
import { CodeToggleBlock } from "../ui/CodeToggleBlock";
import { SimulationHost } from "@/features/simulation/SimulationHost";
import { FIGURE_DEFS } from "@/lib/figure-definitions";
import { markdownToHtml } from "@/lib/markdown-to-html";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

// Error placeholder
function SimulationError({ id, type }: { id: string; type: string }) {
  return (
    <div className="flex h-64 w-full flex-col items-center justify-center rounded-xl border border-[var(--danger-border)] bg-[var(--danger-surface)] text-[var(--text-soft)]">
      <span className="mb-2 text-[var(--accent-strong)]">Unknown placeholder type</span>
      <span className="text-sm">[[{type} {id}]]</span>
    </div>
  );
}

// Render simulation based on type and id
function renderPlaceholder(type: string, id: string): React.ReactNode {
  if (type === "simulation") {
    return <SimulationHost id={id} />;
  }

  if (type === "figure") {
    const knownFigure = FIGURE_DEFS[id];
    if (knownFigure) {
      return (
        <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
          <Image
            src={knownFigure.src}
            alt={knownFigure.caption}
            width={1200}
            height={700}
            className="max-h-[440px] w-full rounded border border-[var(--border-strong)] bg-[var(--surface-1)] object-contain"
          />
          <div className="mt-2 text-xs text-[var(--text-soft)]">{knownFigure.caption}</div>
        </div>
      );
    }

    const lower = id.toLowerCase();
    const hasExt = /\.[a-z0-9]+$/i.test(id);
    const src = hasExt ? `/figures/${id}` : `/figures/${id}.png`;
    const isVideo = lower.endsWith(".mp4") || lower.endsWith(".webm");

    if (isVideo) {
      return (
        <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
          <video controls className="w-full rounded" src={src}>
            Your browser does not support embedded video.
          </video>
          <div className="mt-2 text-xs text-[var(--text-soft)]">Figure: {id}</div>
        </div>
      );
    }

    return (
      <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
        <Image
          src={src}
          alt={`Figure ${id}`}
          width={1200}
          height={700}
          className="w-full rounded border border-[var(--border-strong)]"
        />
        <div className="mt-2 text-xs text-[var(--text-soft)]">Figure: {id}</div>
      </div>
    );
  }

  if (type === "code-toggle") {
    const [pythonCode, pseudoCode] = id.split('|||');
    return <CodeToggleBlock pythonCode={pythonCode} pseudoCode={pseudoCode} />;
  }

  if (type === "code-editor") {
    // Parse the id for parameters: initialCode|expectedOutput|solution
    const [initialCode, expectedOutput, solution] = id.split('|').map(s => s.trim());
    return (
      <CodeEditor
        initialCode={initialCode || ""}
        expectedOutput={expectedOutput}
        solution={solution}
        isExercise={!!expectedOutput}
      />
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
  const [katexLoadFailed, setKatexLoadFailed] = useState(false);

  useEffect(() => {
    import("katex")
      .then((k) => setKatex(k))
      .catch(() => setKatexLoadFailed(true));
  }, []);

  useEffect(() => {
    if (!katex && !katexLoadFailed) return;

    const result = markdownToHtml(content, katex);
    setProcessedContent(result);
  }, [content, katex, katexLoadFailed]);

  if (!processedContent.html) {
    return <div className={className}>Loading...</div>;
  }

  // Split content by placeholder markers and render with components
  const parts = processedContent.html.split(/%%PLACEHOLDER_(\d+)%%/);

  return (
    <div className={`markdown-content prose max-w-none ${className}`}>
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
