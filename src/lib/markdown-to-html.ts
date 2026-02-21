import { FIGURE_DEFS } from "./figure-definitions";
import { SIMULATION_DESCRIPTIONS, simulationTitle } from "./simulation-descriptions";

export type FigureMode = "screenshot" | "alt-text" | "code";

export interface MarkdownToHtmlOptions {
  /** When true, renders interactive elements as static placeholders for PDF export */
  staticMode?: boolean;
  /** When set, figure images use this base path in generated HTML (e.g. "figures/") */
  figureBasePath?: string;
  /** Which figure representations to include in static mode (default: screenshot + alt-text) */
  figureModes?: Set<FigureMode>;
}

export interface MarkdownToHtmlResult {
  html: string;
  placeholders: Array<{ type: string; id: string; index: number }>;
  /** Figures referenced in the content (available when staticMode is true) */
  figures: Array<{ id: string; src: string; caption: string; localFilename: string }>;
}

interface KatexLike {
  renderToString(math: string, options?: { displayMode?: boolean; throwOnError?: boolean }): string;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function resolveAbsoluteUrl(src: string): string {
  if (src.startsWith("http://") || src.startsWith("https://")) return src;
  // For local paths, make them absolute using window.location.origin if available,
  // otherwise leave as-is (server context)
  if (typeof window !== "undefined") return `${window.location.origin}${src}`;
  return src;
}

function getExtensionFromUrl(url: string): string {
  const pathname = url.split("?")[0].split("#")[0];
  const match = pathname.match(/\.([a-z0-9]+)$/i);
  return match ? match[1].toLowerCase() : "png";
}

export function markdownToHtml(
  content: string,
  katexModule: KatexLike | null,
  options: MarkdownToHtmlOptions = {},
): MarkdownToHtmlResult {
  const { staticMode = false, figureBasePath, figureModes } = options;

  let processed = content;
  const placeholders: Array<{ type: string; id: string; index: number }> = [];
  const figures: Array<{ id: string; src: string; caption: string; localFilename: string }> = [];
  let placeholderIndex = 0;

  // Extract and replace placeholders with temporary markers
  processed = processed.replace(
    /\[\[(simulation|figure|code-editor)\s+([^\]]+)\]\]/g,
    (_match, type: string, id: string) => {
      if (staticMode) {
        const trimmedId = id.trim();

        if (type === "simulation") {
          const fModes = figureModes ?? new Set<FigureMode>(["screenshot", "alt-text"]);
          if (fModes.size === 0) return "";
          const title = simulationTitle(trimmedId);
          const description = SIMULATION_DESCRIPTIONS[trimmedId];
          const parts: string[] = [];
          if (fModes.has("screenshot") || fModes.has("alt-text")) {
            parts.push(`<div style="font-size:14px;font-weight:600;margin-bottom:4px;text-align:center;">Interactive Simulation: ${escapeHtml(title)}</div>`);
            if (description) parts.push(`<div style="font-size:12px;margin-top:8px;line-height:1.5;">${escapeHtml(description)}</div>`);
          }
          if (parts.length === 0) return "";
          return `<div style="border:2px dashed #555;border-radius:12px;padding:24px;margin:16px 0;background:#111;color:#999;">${parts.join("\n")}</div>`;
        }

        if (type === "figure") {
          const fModes = figureModes ?? new Set<FigureMode>(["screenshot", "alt-text"]);
          if (fModes.size === 0) return "";
          const knownFigure = FIGURE_DEFS[trimmedId];
          const caption = knownFigure?.caption ?? `Figure: ${trimmedId}`;
          let fetchableSrc: string;
          let localFilename: string;
          if (knownFigure) {
            fetchableSrc = resolveAbsoluteUrl(knownFigure.src);
            const ext = getExtensionFromUrl(knownFigure.src);
            localFilename = `${trimmedId}.${ext}`;
          } else {
            const hasExt = /\.[a-z0-9]+$/i.test(trimmedId);
            const localSrc = hasExt ? `/figures/${trimmedId}` : `/figures/${trimmedId}.png`;
            fetchableSrc = resolveAbsoluteUrl(localSrc);
            localFilename = `${trimmedId}.${hasExt ? localSrc.split(".").pop()! : "png"}`;
          }
          const displaySrc = figureBasePath ? `${figureBasePath}${localFilename}` : fetchableSrc;
          if (fModes.has("screenshot")) {
            figures.push({ id: trimmedId, src: fetchableSrc, caption, localFilename });
          }
          const parts: string[] = [];
          if (fModes.has("screenshot")) {
            parts.push(`<img src="${displaySrc}" alt="${escapeHtml(caption)}" style="max-width:100%;max-height:440px;border-radius:8px;border:1px solid #333;" />`);
          }
          if (fModes.has("alt-text")) {
            parts.push(`<div style="margin-top:8px;font-size:11px;color:#999;">${escapeHtml(caption)}</div>`);
          }
          return `<div style="margin:16px 0;text-align:center;">${parts.join("\n")}</div>`;
        }

        if (type === "code-editor") {
          const fModes = figureModes ?? new Set<FigureMode>(["screenshot", "alt-text"]);
          if (!fModes.has("code")) return "";
          const [initialCode] = trimmedId.split("|").map((s: string) => s.trim());
          return `<pre style="background:#0a0a15;color:#e5e7eb;border-radius:8px;padding:16px;margin:16px 0;overflow-x:auto;font-size:13px;"><code>${escapeHtml(initialCode || "")}</code></pre>`;
        }
      }

      const marker = `%%PLACEHOLDER_${placeholderIndex}%%`;
      placeholders.push({ type, id: id.trim(), index: placeholderIndex });
      placeholderIndex++;
      return marker;
    },
  );

  // Extract code-toggle paired blocks (python + pseudocode) before general code block processing
  processed = processed.replace(
    /```python\n([\s\S]*?)```\s*<!--code-toggle-->\s*```pseudocode\n([\s\S]*?)```/g,
    (_match, pythonCode: string, pseudoCode: string) => {
      if (staticMode) {
        const fModes = figureModes ?? new Set<FigureMode>(["screenshot", "alt-text"]);
        if (!fModes.has("code")) return "";
        return `<pre style="background:#0a0a15;color:#e5e7eb;border-radius:8px;padding:16px;margin:16px 0;overflow-x:auto;font-size:13px;"><code>${escapeHtml(pythonCode.trimEnd())}</code></pre>`;
      }
      const marker = `%%PLACEHOLDER_${placeholderIndex}%%`;
      placeholders.push({
        type: "code-toggle",
        id: `${pythonCode.trimEnd()}|||${pseudoCode.trimEnd()}`,
        index: placeholderIndex,
      });
      placeholderIndex++;
      return marker;
    },
  );

  // Convert block LaTeX ($$...$$ or \[...\])
  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, math: string) => {
    try {
      return `<div class="katex-block">${katexModule ? katexModule.renderToString(math.trim(), { displayMode: true, throwOnError: false }) : math.trim()}</div>`;
    } catch {
      return `$$${math}$$`;
    }
  });

  processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => {
    try {
      return `<div class="katex-block">${katexModule ? katexModule.renderToString(math.trim(), { displayMode: true, throwOnError: false }) : math.trim()}</div>`;
    } catch {
      return `\\[${math}\\]`;
    }
  });

  // Convert inline LaTeX ($...$ or \(...\))
  processed = processed.replace(/\$([^$\n]+?)\$/g, (_, math: string) => {
    try {
      return `<span class="katex-inline">${katexModule ? katexModule.renderToString(math.trim(), { displayMode: false, throwOnError: false }) : math.trim()}</span>`;
    } catch {
      return `$${math}$`;
    }
  });

  processed = processed.replace(/\\\(([^)]+?)\\\)/g, (_, math: string) => {
    try {
      return `<span class="katex-inline">${katexModule ? katexModule.renderToString(math.trim(), { displayMode: false, throwOnError: false }) : math.trim()}</span>`;
    } catch {
      return `\\(${math}\\)`;
    }
  });

  // Convert headers (order matters: longest prefix first)
  processed = processed.replace(/^###### (.+)$/gm, '<h6 class="text-sm font-semibold mt-4 mb-1 text-white">$1</h6>');
  processed = processed.replace(/^##### (.+)$/gm, '<h5 class="text-sm font-semibold mt-4 mb-1 text-white">$1</h5>');
  processed = processed.replace(/^#### (.+)$/gm, '<h4 class="text-base font-semibold mt-5 mb-2 text-white">$1</h4>');
  processed = processed.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-2 text-white">$1</h3>');
  processed = processed.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-3 text-white">$1</h2>');
  processed = processed.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-4 text-white">$1</h1>');

  // Convert bold and italic
  processed = processed.replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>");
  processed = processed.replace(/__([\s\S]+?)__/g, "<strong>$1</strong>");
  processed = processed.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Convert links
  processed = processed.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" class="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>',
  );

  // Convert code blocks
  processed = processed.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    '<pre class="bg-[#0a0a15] text-gray-100 rounded-lg p-4 my-4 overflow-x-auto text-sm"><code>$2</code></pre>',
  );

  // Convert inline code
  processed = processed.replace(
    /`([^`]+)`/g,
    '<code class="bg-[#0a0a15] px-1.5 py-0.5 rounded text-sm font-mono text-blue-300">$1</code>',
  );

  // Convert lists
  processed = processed.replace(/^- (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');
  processed = processed.replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc my-4 space-y-1">$&</ul>');

  // Convert numbered lists
  processed = processed.replace(/^\d+\. (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');

  // Convert tables
  processed = processed.replace(/^\|(.+)\|$/gm, (_match, content: string) => {
    const cells = content.split("|").map((c: string) => c.trim());
    const isHeader = cells.some((c: string) => c.match(/^-+$/));
    if (isHeader) return "";
    const tag = "td";
    return `<tr>${cells.map((c: string) => `<${tag} class="border border-gray-700 px-3 py-2 text-gray-300">${c}</${tag}>`).join("")}</tr>`;
  });

  // Wrap tables
  processed = processed.replace(
    /(<tr>.*<\/tr>\s*)+/g,
    '<table class="w-full border-collapse my-4 bg-[#0a0a15] rounded">$&</table>',
  );

  // Add table headers (first row after <table>)
  processed = processed.replace(/<table([^>]*)>(<tr>.*?<\/tr>)/g, (_match, attrs: string, firstRow: string) => {
    const headerRow = firstRow.replace(/<td/g, "<th").replace(/<\/td>/g, "</th>");
    return `<table${attrs}>${headerRow}`;
  });

  // Convert horizontal rules
  processed = processed.replace(/^---$/gm, '<hr class="border-gray-700 my-8" />');

  // Convert paragraphs (double newlines)
  processed = processed
    .split("\n\n")
    .map((para) => {
      if (para.trim() && !para.startsWith("<")) {
        return `<p class="my-4 text-gray-300 leading-relaxed">${para}</p>`;
      }
      return para;
    })
    .join("\n");

  return { html: processed, placeholders, figures };
}
