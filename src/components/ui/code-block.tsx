"use client";

import { useState, useEffect } from "react";

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
}

export function CodeBlock({ 
  code, 
  language = "typescript", 
  filename,
  showLineNumbers = true 
}: CodeBlockProps) {
  const [html, setHtml] = useState<string>("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    // Dynamic import for shiki
    import("shiki").then(async (shiki) => {
      const highlighter = await shiki.createHighlighter({
        themes: ["github-dark", "github-light"],
        langs: ["typescript", "python", "javascript", "jsx", "tsx", "json", "bash", "markdown", "latex"],
      });

      const highlighted = highlighter.codeToHtml(code, {
        lang: language,
        themes: {
          light: "github-light",
          dark: "github-dark",
        },
      });

      setHtml(highlighted);
    });
  }, [code, language]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = code.split("\n");

  return (
    <div className="relative my-4 overflow-hidden rounded-lg border border-gray-200 bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-700 bg-gray-800 px-4 py-2">
        <div className="flex items-center gap-2">
          {filename ? (
            <span className="text-sm font-medium text-gray-300">{filename}</span>
          ) : (
            <span className="text-sm text-gray-400">{language}</span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="rounded px-2 py-1 text-xs text-gray-400 hover:bg-gray-700 hover:text-gray-200"
        >
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>

      {/* Code */}
      <div className="relative">
        {showLineNumbers && (
          <div className="pointer-events-none absolute left-0 top-0 flex select-none flex-col py-4 pl-4 pr-2 text-right text-gray-500">
            {lines.map((_, i) => (
              <span key={i} className="text-xs leading-6">
                {i + 1}
              </span>
            ))}
          </div>
        )}
        <div 
          className={`overflow-x-auto py-4 ${showLineNumbers ? "pl-12 pr-4" : "px-4"}`}
        >
          {html ? (
            <div 
              className="[&_.shiki]:!bg-transparent [&_.shiki]:text-sm [&_pre]:!bg-transparent"
              dangerouslySetInnerHTML={{ __html: html }} 
            />
          ) : (
            <pre className="text-sm text-gray-300">
              <code>{code}</code>
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

// Simple inline code component
export function InlineCode({ children }: { children: string }) {
  return (
    <code className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-sm text-gray-800 dark:bg-gray-800 dark:text-gray-200">
      {children}
    </code>
  );
}
