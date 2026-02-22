import Link from "next/link";
import { ExportPdfButton } from "@/components/content/ExportPdfButton";
import { topicHref } from "@/lib/topic-navigation";

interface TopicSidebarProps {
  routeSlug: string;
  topicMeta: { title: string; description: string };
  topicId: string;
  lessons: Array<{ slug: string; title: string; summary?: string }>;
  activeSlug?: string;
}

export function TopicSidebar({ routeSlug, topicMeta, topicId, lessons, activeSlug }: TopicSidebarProps) {
  return (
    <aside className="h-fit rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-4">
      <div className="border-b border-[var(--border-strong)] pb-3">
        <div className="text-[11px] uppercase tracking-[0.18em] text-[var(--accent)]">Topic</div>
        <Link href={topicHref(routeSlug)} className="mt-1 block text-lg font-semibold text-[var(--text-strong)] hover:text-[var(--accent)] transition">{topicMeta.title}</Link>
        <p className="mt-2 text-xs text-[var(--text-muted)]">{topicMeta.description}</p>
        <ExportPdfButton topicId={topicId} topicTitle={topicMeta.title} />
      </div>

      <div className="mt-4 space-y-1 font-mono text-sm">
        {lessons.map((entry) => {
          const active = entry.slug === activeSlug;
          return (
            <Link
              key={entry.slug}
              href={topicHref(routeSlug, entry.slug)}
              title={entry.summary}
              className={`block rounded-md border px-3 py-2 transition ${
                active
                  ? "border-[var(--accent)] bg-[var(--surface-3)] text-[var(--accent-strong)]"
                  : "border-transparent text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:bg-[var(--surface-2)] hover:text-[var(--text-strong)]"
              }`}
            >
              <span className="text-[var(--accent)]">$</span> {entry.title}
              {entry.summary && (
                <span className="mt-0.5 block font-sans text-[10px] leading-tight text-[var(--text-soft)]">
                  {entry.summary}
                </span>
              )}
            </Link>
          );
        })}
      </div>
    </aside>
  );
}
