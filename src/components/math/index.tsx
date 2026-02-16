"use client";

import { useEffect, useState } from "react";

interface MathProps {
  children: string;
}

// Dynamic import for KaTeX to avoid SSR issues
function KaTeXRenderer({ math, displayMode }: { math: string; displayMode: boolean }) {
  const [html, setHtml] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    import("katex").then((katex) => {
      try {
        const rendered = katex.renderToString(math, {
          displayMode,
          throwOnError: false,
          strict: false,
        });
        setHtml(rendered);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : "KaTeX error");
      }
    });
  }, [math, displayMode]);

  if (error) {
    return <span className="text-red-500">{error}</span>;
  }

  if (!html) {
    return <span className="animate-pulse">Loading...</span>;
  }

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export function InlineMath({ children }: MathProps) {
  return (
    <span className="inline">
      <KaTeXRenderer math={children} displayMode={false} />
    </span>
  );
}

export function BlockMath({ children }: MathProps) {
  return (
    <div className="my-4 overflow-x-auto text-center">
      <KaTeXRenderer math={children} displayMode={true} />
    </div>
  );
}

export function Equation({ children, number }: MathProps & { number?: string }) {
  return (
    <div className="my-4 flex items-center justify-center gap-4 overflow-x-auto">
      <KaTeXRenderer math={children} displayMode={true} />
      {number && <span className="text-sm text-gray-500">({number})</span>}
    </div>
  );
}
