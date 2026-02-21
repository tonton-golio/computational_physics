"use client";

import { useState } from "react";
import { markdownToHtml } from "@/lib/markdown-to-html";

type ExportState = "idle" | "loading" | "error";

interface ExportPdfButtonProps {
  topicContentId: string;
  topicTitle: string;
  /** "default" renders full-width sidebar button; "compact" renders a small icon button for cards */
  variant?: "default" | "compact";
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function sanitizeFilename(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

const BASE_CSS = `
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  color: #1a1a1a;
  background: #fff;
  padding: 0;
  font-size: 14px;
  line-height: 1.6;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { color: #111; margin-top: 1.4em; margin-bottom: 0.5em; }
h1 { font-size: 20px; font-weight: 700; }
h2 { font-size: 18px; font-weight: 700; }
h3 { font-size: 16px; font-weight: 600; }
h4 { font-size: 15px; font-weight: 600; }
h5, h6 { font-size: 14px; font-weight: 600; }

p { color: #333; line-height: 1.7; margin: 10px 0; }
strong { color: #111; }
em { font-style: italic; }
a { color: #2563eb; text-decoration: underline; }

/* Lists */
ul { list-style: disc; margin: 10px 0; padding-left: 24px; }
ol { list-style: decimal; margin: 10px 0; padding-left: 24px; }
li { color: #333; margin: 4px 0; }

/* Code */
pre {
  background: #f3f4f6 !important;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 14px;
  overflow-x: auto;
  font-size: 12px;
  color: #1a1a1a;
  white-space: pre-wrap;
  word-wrap: break-word;
  margin: 12px 0;
}
code {
  font-family: 'Fira Code', 'Cascadia Code', Consolas, monospace;
  font-size: 12px;
}
:not(pre) > code {
  background: #f3f4f6;
  padding: 1px 5px;
  border-radius: 3px;
  color: #c026d3;
}

/* Tables */
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; color: #333; }
th { background: #f9fafb; font-weight: 600; color: #111; }

/* KaTeX */
.katex-block { text-align: center; margin: 14px 0; overflow-x: auto; }
.katex-inline { display: inline; }

/* Images & figures */
img { max-width: 100%; height: auto; border-radius: 4px; }

/* Horizontal rules */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }

/* Print styles */
@media print {
  body { padding: 0; }
  pre, table, img { page-break-inside: avoid; }
}

@media screen {
  body { max-width: 794px; margin: 0 auto; padding: 40px 32px; }
}
`;

function buildLessonHtml(
  title: string,
  bodyHtml: string,
  topicTitle: string,
  katexCss: string,
): string {
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>${escapeHtml(title)} — ${escapeHtml(topicTitle)} — KoalaBrain</title>
<style>
${katexCss}
${BASE_CSS}

.lesson-header {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 2px solid #e5e7eb;
}
.lesson-header .topic-label {
  font-size: 11px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #888;
  margin-bottom: 6px;
}
.lesson-header h1 {
  font-size: 24px;
  font-weight: 800;
  color: #111;
  margin: 0;
}
</style>
</head>
<body>
  <div class="lesson-header">
    <div class="topic-label">${escapeHtml(topicTitle)}</div>
    <h1>${escapeHtml(title)}</h1>
  </div>
  <div class="lesson-content">${bodyHtml}</div>
</body>
</html>`;
}

function buildContentsHtml(
  topicTitle: string,
  lessons: Array<{ index: number; title: string; summary?: string; filename: string }>,
  dateStr: string,
): string {
  const rows = lessons
    .map(
      (l) => `
      <tr>
        <td style="white-space:nowrap;color:#888;font-size:12px;vertical-align:top;padding-right:12px;">${String(l.index).padStart(2, "0")}</td>
        <td>
          <a href="${l.filename}" style="font-weight:600;color:#111;text-decoration:none;">${escapeHtml(l.title)}</a>
          ${l.summary ? `<div style="font-size:12px;color:#666;margin-top:2px;">${escapeHtml(l.summary)}</div>` : ""}
        </td>
      </tr>`,
    )
    .join("\n");

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>${escapeHtml(topicTitle)} — Contents — KoalaBrain</title>
<style>
${BASE_CSS}

.cover {
  text-align: center;
  padding: 60px 0 40px;
  border-bottom: 2px solid #e5e7eb;
  margin-bottom: 32px;
}
.cover .brand {
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #888;
  margin-bottom: 16px;
}
.cover h1 {
  font-size: 32px;
  font-weight: 800;
  color: #111;
  margin-bottom: 8px;
}
.cover .meta {
  font-size: 13px;
  color: #888;
}

.toc-heading {
  font-size: 14px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #888;
  margin-bottom: 16px;
}

.toc-table {
  width: 100%;
  border-collapse: collapse;
}
.toc-table td {
  border: none;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
}
.toc-table tr:last-child td {
  border-bottom: none;
}
</style>
</head>
<body>
  <div class="cover">
    <div class="brand">KoalaBrain</div>
    <h1>${escapeHtml(topicTitle)}</h1>
    <div class="meta">${lessons.length} lessons &middot; ${dateStr}</div>
  </div>
  <div class="toc-heading">Contents</div>
  <table class="toc-table">
    ${rows}
  </table>
</body>
</html>`;
}

export function ExportPdfButton({ topicContentId, topicTitle, variant = "default" }: ExportPdfButtonProps) {
  const [state, setState] = useState<ExportState>("idle");

  async function handleExport() {
    setState("loading");

    try {
      const [res, JSZip, katex] = await Promise.all([
        fetch(`/api/content/export?topic=${encodeURIComponent(topicContentId)}`),
        import("jszip").then((m) => m.default),
        import("katex"),
      ]);

      if (!res.ok) throw new Error("Failed to fetch lessons");
      const { lessons } = (await res.json()) as {
        lessons: Array<{ slug: string; title: string; content: string; summary?: string }>;
      };

      if (!lessons.length) throw new Error("No lessons found");

      // Fetch KaTeX CSS for math rendering in HTML files
      const katexCdnBase = "https://cdn.jsdelivr.net/npm/katex@0.16.28/dist";
      let katexCss = "";
      try {
        const cssRes = await fetch(`${katexCdnBase}/katex.min.css`);
        if (cssRes.ok) {
          katexCss = (await cssRes.text()).replace(
            /url\(fonts\//g,
            `url(${katexCdnBase}/fonts/`,
          );
        }
      } catch {
        // Non-fatal
      }

      const dateStr = new Date().toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });

      const zip = new JSZip();
      const topicFolder = sanitizeFilename(topicTitle);

      // Build individual lesson files and collect TOC entries + figures
      const tocEntries: Array<{ index: number; title: string; summary?: string; filename: string }> = [];
      const allFigures = new Map<string, { src: string; caption: string }>();

      for (let i = 0; i < lessons.length; i++) {
        const lesson = lessons[i];
        const { html, figures } = markdownToHtml(lesson.content, katex, {
          staticMode: true,
          figureBasePath: "figures/",
        });
        const num = String(i + 1).padStart(2, "0");
        const filename = `${num}-${sanitizeFilename(lesson.title)}.html`;

        tocEntries.push({
          index: i + 1,
          title: lesson.title,
          summary: lesson.summary,
          filename,
        });

        for (const fig of figures) {
          if (!allFigures.has(fig.localFilename)) {
            allFigures.set(fig.localFilename, { src: fig.src, caption: fig.caption });
          }
        }

        const lessonDoc = buildLessonHtml(lesson.title, html, topicTitle, katexCss);
        zip.file(`${topicFolder}/${filename}`, lessonDoc);
      }

      // Fetch and bundle figure images (skip videos)
      const VIDEO_EXTS = new Set(["mp4", "webm", "ogg", "avi", "mov"]);
      const figureEntries = Array.from(allFigures.entries()).filter(
        ([filename]) => !VIDEO_EXTS.has(filename.split(".").pop()?.toLowerCase() ?? ""),
      );

      const fetchedFigures = await Promise.all(
        figureEntries.map(async ([filename, { src }]) => {
          try {
            const response = await fetch(src);
            if (!response.ok) return null;
            const blob = await response.blob();
            return { filename, blob };
          } catch {
            return null;
          }
        }),
      );

      for (const result of fetchedFigures) {
        if (result) {
          zip.file(`${topicFolder}/figures/${result.filename}`, result.blob);
        }
      }

      // Build table of contents file
      const contentsHtml = buildContentsHtml(topicTitle, tocEntries, dateStr);
      zip.file(`${topicFolder}/00-contents.html`, contentsHtml);

      // Generate and download the ZIP
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${topicFolder}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setState("idle");
    } catch (err) {
      console.error("PDF export failed:", err);
      setState("error");
      setTimeout(() => setState("idle"), 3000);
    }
  }

  const downloadIcon = (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );

  if (variant === "compact") {
    return (
      <button
        onClick={handleExport}
        disabled={state === "loading"}
        title={state === "error" ? "Export failed — try again" : "Export ZIP"}
        className="shrink-0 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] p-1.5 text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-3)] hover:text-[var(--text-strong)] disabled:cursor-wait disabled:opacity-60"
      >
        {state === "loading" ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
          </svg>
        ) : (
          downloadIcon
        )}
      </button>
    );
  }

  return (
    <button
      onClick={handleExport}
      disabled={state === "loading"}
      className="mt-3 flex w-full items-center justify-center gap-2 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-2 text-xs font-medium text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-3)] hover:text-[var(--text-strong)] disabled:cursor-wait disabled:opacity-60"
    >
      {state === "idle" && (
        <>
          {downloadIcon}
          Export ZIP
        </>
      )}
      {state === "loading" && "Preparing export..."}
      {state === "error" && "Export failed — try again"}
    </button>
  );
}
