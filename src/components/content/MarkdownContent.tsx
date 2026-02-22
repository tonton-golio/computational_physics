"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { CodeEditor } from "@/components/ui/code-editor";
import { CodeToggleBlock } from "@/components/ui/CodeToggleBlock";
import { SimulationHost } from "@/features/simulation/SimulationHost";
import { FIGURE_DEFS } from "@/lib/figure-definitions";
import { markdownToHtml } from "@/lib/markdown-to-html";
import { FigureLightbox } from "@/components/ui/figure-lightbox";

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

// Reusable figure wrapper with lightbox expand support
function FigureBlock({
  src,
  alt,
  caption,
  isVideo,
  children,
}: {
  src: string;
  alt: string;
  caption: string;
  isVideo?: boolean;
  children: React.ReactNode;
}) {
  const [lightboxOpen, setLightboxOpen] = useState(false);

  return (
    <>
      <div className="group relative w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
        <div
          className={isVideo ? undefined : "cursor-zoom-in"}
          onClick={isVideo ? undefined : () => setLightboxOpen(true)}
        >
          {children}
        </div>
        <div className="mt-2 text-xs text-[var(--text-soft)]">{caption}</div>

        {!isVideo && (
          <button
            type="button"
            onClick={() => setLightboxOpen(true)}
            aria-label="View full size"
            className="absolute top-6 right-6 flex h-8 w-8 items-center justify-center rounded-md bg-[var(--surface-2)]/70 backdrop-blur-sm border border-[var(--border-strong)] text-[var(--text-soft)] hover:text-[var(--text-strong)] hover:bg-[var(--surface-3)]/80 transition-all opacity-0 group-hover:opacity-100 cursor-pointer"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          </button>
        )}
      </div>

      {lightboxOpen && (
        <FigureLightbox
          src={src}
          alt={alt}
          isVideo={isVideo}
          caption={caption}
          onClose={() => setLightboxOpen(false)}
        />
      )}
    </>
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
        <FigureBlock src={knownFigure.src} alt={knownFigure.caption} caption={knownFigure.caption}>
          <Image
            src={knownFigure.src}
            alt={knownFigure.caption}
            width={1200}
            height={700}
            className="max-h-[440px] w-full rounded border border-[var(--border-strong)] bg-[var(--surface-1)] object-contain"
          />
        </FigureBlock>
      );
    }

    const lower = id.toLowerCase();
    const hasExt = /\.[a-z0-9]+$/i.test(id);
    const src = hasExt ? `/figures/${id}` : `/figures/${id}.png`;
    const isVideo = lower.endsWith(".mp4") || lower.endsWith(".webm");

    if (isVideo) {
      return (
        <FigureBlock src={src} alt={`Figure ${id}`} caption={`Figure: ${id}`} isVideo>
          <video controls className="w-full rounded" src={src}>
            Your browser does not support embedded video.
          </video>
        </FigureBlock>
      );
    }

    return (
      <FigureBlock src={src} alt={`Figure ${id}`} caption={`Figure: ${id}`}>
        <Image
          src={src}
          alt={`Figure ${id}`}
          width={1200}
          height={700}
          className="w-full rounded border border-[var(--border-strong)]"
        />
      </FigureBlock>
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
