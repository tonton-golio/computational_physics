"use client";

import { useState, useEffect } from "react";
import { markdownToHtml, type FigureMode } from "@/lib/markdown-to-html";

/* ─── Types ─── */

type ExportState = "idle" | "fetching" | "ready" | "exporting" | "error";
type ExportFormat = "zip" | "pdf";

interface Lesson {
  slug: string;
  title: string;
  content: string;
  summary?: string;
}

interface ExportPdfButtonProps {
  topicContentId: string;
  topicTitle: string;
  /** "default" renders full-width sidebar button; "compact" renders a small icon button for cards */
  variant?: "default" | "compact";
}

/* ─── Helpers ─── */

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function sanitizeFilename(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/* ─── Styles ─── */

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

/* ─── HTML builders ─── */

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

function buildCombinedPdfHtml(
  topicTitle: string,
  lessonHtmls: Array<{ title: string; html: string }>,
  katexCss: string,
): string {
  const lessonsMarkup = lessonHtmls
    .map(
      (l, i) => `
    <div class="lesson"${i > 0 ? ' style="page-break-before:always;"' : ""}>
      <div class="lesson-header">
        <h2 style="font-size:22px;font-weight:800;margin:0;">${escapeHtml(l.title)}</h2>
      </div>
      <div class="lesson-content">${l.html}</div>
    </div>`,
    )
    .join("\n");

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>${escapeHtml(topicTitle)} — KoalaBrain</title>
<style>
${katexCss}
${BASE_CSS}

@page { margin: 2cm; }
body { max-width: none; padding: 0; }
.cover {
  text-align: center;
  padding: 120px 0 60px;
  page-break-after: always;
}
.cover .brand {
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #888;
}
.cover h1 {
  font-size: 36px;
  font-weight: 800;
  margin: 16px 0 8px;
  color: #111;
}
.cover .meta {
  font-size: 14px;
  color: #888;
}
.lesson {
  padding: 0 32px;
}
.lesson-header {
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid #e5e7eb;
}
@media screen {
  body { max-width: 794px; margin: 0 auto; }
  .cover { padding: 80px 0 40px; }
}
</style>
<script>window.addEventListener('load', function(){ setTimeout(function(){ window.print(); }, 500); });<\/script>
</head>
<body>
  <div class="cover">
    <div class="brand">KoalaBrain</div>
    <h1>${escapeHtml(topicTitle)}</h1>
    <div class="meta">${lessonHtmls.length} lessons</div>
  </div>
  ${lessonsMarkup}
</body>
</html>`;
}

/* ─── KaTeX CSS loader ─── */

async function loadKatexCss(): Promise<string> {
  const katexCdnBase = "https://cdn.jsdelivr.net/npm/katex@0.16.28/dist";
  try {
    const res = await fetch(`${katexCdnBase}/katex.min.css`);
    if (!res.ok) return "";
    return (await res.text()).replace(/url\(fonts\//g, `url(${katexCdnBase}/fonts/`);
  } catch {
    return "";
  }
}

/* ─── Export functions ─── */

async function exportAsZip(
  lessons: Lesson[],
  topicTitle: string,
  fModes: Set<FigureMode>,
): Promise<void> {
  const [JSZip, katex, katexCss] = await Promise.all([
    import("jszip").then((m) => m.default),
    import("katex"),
    loadKatexCss(),
  ]);

  const dateStr = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const zip = new JSZip();
  const topicFolder = sanitizeFilename(topicTitle);
  const tocEntries: Array<{ index: number; title: string; summary?: string; filename: string }> = [];
  const allFigures = new Map<string, { src: string; caption: string }>();

  for (let i = 0; i < lessons.length; i++) {
    const lesson = lessons[i];
    const { html, figures } = markdownToHtml(lesson.content, katex, {
      staticMode: true,
      figureBasePath: "figures/",
      figureModes: fModes,
    });
    const num = String(i + 1).padStart(2, "0");
    const filename = `${num}-${sanitizeFilename(lesson.title)}.html`;

    tocEntries.push({ index: i + 1, title: lesson.title, summary: lesson.summary, filename });

    for (const fig of figures) {
      if (!allFigures.has(fig.localFilename)) {
        allFigures.set(fig.localFilename, { src: fig.src, caption: fig.caption });
      }
    }

    zip.file(`${topicFolder}/${filename}`, buildLessonHtml(lesson.title, html, topicTitle, katexCss));
  }

  // Fetch figure images (only populated when "screenshot" is in figureModes)
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
    if (result) zip.file(`${topicFolder}/figures/${result.filename}`, result.blob);
  }

  zip.file(`${topicFolder}/00-contents.html`, buildContentsHtml(topicTitle, tocEntries, dateStr));

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${topicFolder}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function exportAsPdf(
  lessons: Lesson[],
  topicTitle: string,
  fModes: Set<FigureMode>,
): Promise<void> {
  const [katex, katexCss] = await Promise.all([import("katex"), loadKatexCss()]);

  const lessonHtmls = lessons.map((lesson) => {
    const { html } = markdownToHtml(lesson.content, katex, {
      staticMode: true,
      figureModes: fModes,
    });
    return { title: lesson.title, html };
  });

  const combinedHtml = buildCombinedPdfHtml(topicTitle, lessonHtmls, katexCss);

  const win = window.open("", "_blank");
  if (!win) throw new Error("Popup blocked — please allow popups for this site.");
  win.document.write(combinedHtml);
  win.document.close();
}

/* ─── Modal ─── */

const FIGURE_MODE_OPTIONS: Array<{ mode: FigureMode; label: string; desc: string }> = [
  { mode: "screenshot", label: "Screenshots", desc: "Embed figure images" },
  { mode: "alt-text", label: "Alt text", desc: "Include figure captions" },
  { mode: "code", label: "Code", desc: "Include code blocks" },
];

function ExportModal({
  topicContentId,
  topicTitle,
  onClose,
}: {
  topicContentId: string;
  topicTitle: string;
  onClose: () => void;
}) {
  const [state, setState] = useState<ExportState>("fetching");
  const [lessons, setLessons] = useState<Lesson[]>([]);
  const [selectedSlugs, setSelectedSlugs] = useState<Set<string>>(new Set());
  const [format, setFormat] = useState<ExportFormat>("zip");
  const [fModes, setFModes] = useState<Set<FigureMode>>(new Set(["screenshot", "alt-text", "code"]));

  useEffect(() => {
    fetch(`/api/content/export?topic=${encodeURIComponent(topicContentId)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed");
        return res.json();
      })
      .then((data: { lessons: Lesson[] }) => {
        setLessons(data.lessons);
        setSelectedSlugs(new Set(data.lessons.map((l) => l.slug)));
        setState("ready");
      })
      .catch(() => setState("error"));
  }, [topicContentId]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  const PINNED_SLUGS = new Set(["home", "intro", "introduction", "landingPage"]);
  const isPinned = (slug: string) => PINNED_SLUGS.has(slug);

  const toggleLesson = (slug: string) => {
    if (isPinned(slug)) return;
    setSelectedSlugs((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) next.delete(slug);
      else next.add(slug);
      return next;
    });
  };

  const toggleFigureMode = (mode: FigureMode) => {
    setFModes((prev) => {
      const next = new Set(prev);
      if (next.has(mode)) next.delete(mode);
      else next.add(mode);
      return next;
    });
  };

  const handleExport = async () => {
    const selected = lessons.filter((l) => selectedSlugs.has(l.slug));
    if (!selected.length) return;
    setState("exporting");
    try {
      if (format === "zip") {
        await exportAsZip(selected, topicTitle, fModes);
      } else {
        await exportAsPdf(selected, topicTitle, fModes);
      }
      onClose();
    } catch (err) {
      console.error("Export failed:", err);
      setState("error");
      setTimeout(() => setState("ready"), 3000);
    }
  };

  const selectedCount = selectedSlugs.size;
  const allSelected = selectedCount === lessons.length;
  const canExport = (state === "ready" || state === "error") && selectedCount > 0;

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="mx-4 flex max-h-[85vh] w-full max-w-md flex-col rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[var(--border-strong)] px-5 py-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-soft)]">Export</div>
            <div className="text-sm font-semibold text-[var(--text-strong)]">{topicTitle}</div>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-[var(--text-soft)] transition hover:text-[var(--text-strong)]"
            aria-label="Close"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M4 4l8 8M12 4l-8 8" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
          {state === "fetching" && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-pulse text-xs text-[var(--text-soft)]">Loading lessons…</div>
            </div>
          )}

          {state === "error" && lessons.length === 0 && (
            <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-surface)] px-4 py-3 text-xs text-[var(--text-strong)]">
              Failed to load lessons. Please try again.
            </div>
          )}

          {lessons.length > 0 && (
            <>
              {/* Format selection */}
              <div>
                <div className="mb-2 text-[10px] uppercase tracking-[0.15em] text-[var(--text-soft)]">Format</div>
                <div className="flex gap-2">
                  {(["zip", "pdf"] as const).map((f) => (
                    <button
                      key={f}
                      onClick={() => setFormat(f)}
                      className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                        format === f
                          ? "border-[var(--accent)] bg-[var(--accent)]/10 text-[var(--accent-strong)]"
                          : "border-[var(--border-strong)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
                      }`}
                    >
                      {f === "zip" ? "ZIP (HTML files)" : "PDF (printable)"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Lesson selection */}
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-soft)]">
                    Lessons ({selectedCount}/{lessons.length})
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setSelectedSlugs(new Set(lessons.map((l) => l.slug)))}
                      disabled={allSelected}
                      className="text-[10px] text-[var(--accent)] transition hover:text-[var(--accent-strong)] disabled:opacity-30"
                    >
                      All
                    </button>
                    <button
                      onClick={() => setSelectedSlugs(new Set(lessons.filter((l) => isPinned(l.slug)).map((l) => l.slug)))}
                      disabled={selectedCount === 0}
                      className="text-[10px] text-[var(--accent)] transition hover:text-[var(--accent-strong)] disabled:opacity-30"
                    >
                      None
                    </button>
                  </div>
                </div>
                <div className="max-h-48 overflow-y-auto rounded-lg border border-[var(--border-strong)] bg-[var(--surface-2)]">
                  {lessons.map((lesson) => {
                    const pinned = isPinned(lesson.slug);
                    return (
                      <label
                        key={lesson.slug}
                        className={`flex items-center gap-2.5 border-b border-[var(--border-strong)] px-3 py-2 text-xs transition last:border-b-0 ${pinned ? "opacity-70" : "cursor-pointer hover:bg-[var(--surface-3)]"}`}
                      >
                        {pinned ? (
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0 text-[var(--text-soft)]">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                            <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                          </svg>
                        ) : (
                          <input
                            type="checkbox"
                            checked={selectedSlugs.has(lesson.slug)}
                            onChange={() => toggleLesson(lesson.slug)}
                            className="h-3.5 w-3.5 accent-[var(--accent)]"
                          />
                        )}
                        <span
                          className={
                            selectedSlugs.has(lesson.slug)
                              ? "text-[var(--text-strong)]"
                              : "text-[var(--text-muted)]"
                          }
                        >
                          {lesson.title}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>

              {/* Figure modes */}
              <div>
                <div className="mb-2 text-[10px] uppercase tracking-[0.15em] text-[var(--text-soft)]">
                  Include figures as
                </div>
                <div className="space-y-1">
                  {FIGURE_MODE_OPTIONS.map(({ mode, label, desc }) => (
                    <label
                      key={mode}
                      className="flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-2 text-xs transition hover:bg-[var(--surface-2)]"
                    >
                      <input
                        type="checkbox"
                        checked={fModes.has(mode)}
                        onChange={() => toggleFigureMode(mode)}
                        className="h-3.5 w-3.5 accent-[var(--accent)]"
                      />
                      <div>
                        <span
                          className={
                            fModes.has(mode)
                              ? "font-medium text-[var(--text-strong)]"
                              : "text-[var(--text-muted)]"
                          }
                        >
                          {label}
                        </span>
                        <span className="ml-1.5 text-[var(--text-soft)]">{desc}</span>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-[var(--border-strong)] px-5 py-3">
          <button
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={!canExport}
            className="rounded-md border border-[var(--accent)] bg-[var(--accent)]/10 px-4 py-1.5 text-xs font-medium text-[var(--accent-strong)] transition hover:bg-[var(--accent)]/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {state === "exporting" ? "Exporting…" : `Export ${format.toUpperCase()}`}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ─── Entry point ─── */

export function ExportPdfButton({ topicContentId, topicTitle, variant = "default" }: ExportPdfButtonProps) {
  const [showModal, setShowModal] = useState(false);

  const downloadIcon = (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );

  return (
    <>
      {variant === "compact" ? (
        <button
          onClick={() => setShowModal(true)}
          title="Export"
          className="shrink-0 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] p-1.5 text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-3)] hover:text-[var(--text-strong)]"
        >
          {downloadIcon}
        </button>
      ) : (
        <button
          onClick={() => setShowModal(true)}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-2 text-xs font-medium text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-3)] hover:text-[var(--text-strong)]"
        >
          {downloadIcon}
          Export
        </button>
      )}

      {showModal && (
        <ExportModal
          topicContentId={topicContentId}
          topicTitle={topicTitle}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
}
