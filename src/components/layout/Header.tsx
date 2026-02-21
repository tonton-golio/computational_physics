 "use client";

import { Suspense } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { LeaderboardButton } from "@/components/layout/LeaderboardButton";
import { AuthButton } from "@/components/layout/AuthButton";
import { TOPICS } from "@/lib/topic-config";
import { TOPIC_ROUTES } from "@/lib/topic-navigation";

const TOPIC_TITLES: Record<string, string> = Object.fromEntries(
  TOPIC_ROUTES.map(({ routeSlug, contentId }) => [routeSlug, TOPICS[contentId].title])
);

export function Header() {
  return (
    <Suspense
      fallback={
        <header className="sticky top-0 z-50 bg-transparent">
          <div className="flex h-16 w-full items-center gap-3 px-6">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-semibold text-[var(--text-strong)]">koala brain</span>
            </Link>
            <div className="flex-1" />
            <nav className="flex items-center gap-2 shrink-0">
              <AuthButton />
              <LeaderboardButton />
              <ThemeToggle />
            </nav>
          </div>
        </header>
      }
    >
      <HeaderContent />
    </Suspense>
  );
}

function HeaderContent() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const isTopics = pathname === "/topics";
  const isTopicSubpage = !isTopics && pathname.startsWith("/topics/");
  const topicTitle = isTopicSubpage ? TOPIC_TITLES[pathname.split("/")[2]] ?? null : null;
  const view = searchParams.get("view") === "points" ? "points" : "boxes";
  const query = searchParams.get("q") ?? "";

  function updateTopicsParam(key: "q" | "view", value: string | null) {
    if (!isTopics) return;
    const params = new URLSearchParams(searchParams.toString());
    if (!value) {
      params.delete(key);
    } else {
      params.set(key, value);
    }
    const queryString = params.toString();
    router.replace(queryString ? `${pathname}?${queryString}` : pathname, { scroll: false });
  }

  return (
    <header className="sticky top-0 z-50 bg-transparent">
      <div className="relative flex h-16 w-full items-center gap-3 px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-semibold text-[var(--text-strong)]">koala brain</span>
        </Link>

        {isTopicSubpage ? (
          <>
            {/* Centered group: back button + topic title */}
            {topicTitle && (
              <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2">
                <Link
                  href="/topics"
                  className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-3 py-1 text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:text-[var(--accent)]"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 12H5"/><path d="m12 19-7-7 7-7"/></svg>
                  back to topics
                </Link>
                <Link
                  href={`/topics/${pathname.split("/")[2]}`}
                  className="text-sm font-semibold text-[var(--text-strong)] transition hover:text-[var(--accent)]"
                >
                  {topicTitle}
                </Link>
              </div>
            )}
            <div className="flex-1" />
          </>
        ) : isTopics ? (
          <div className="flex min-w-0 flex-1 justify-center">
            <div className="flex w-full max-w-[720px] items-center gap-3">
              <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={() => updateTopicsParam("view", "boxes")}
                className={`rounded-md border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition ${
                  view === "boxes"
                    ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--accent)]"
                    : "border-[var(--border-strong)] bg-[var(--surface-1)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                }`}
              >
                boxes
              </button>
              <button
                type="button"
                onClick={() => updateTopicsParam("view", "points")}
                className={`rounded-md border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition ${
                  view === "points"
                    ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--accent)]"
                    : "border-[var(--border-strong)] bg-[var(--surface-1)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                }`}
              >
                points
              </button>
              </div>

              <label htmlFor="header-topics-search" className="sr-only">
                Search subtopics
              </label>
              <input
                id="header-topics-search"
                type="search"
                value={query}
                onChange={(event) => updateTopicsParam("q", event.target.value.trim() ? event.target.value : null)}
                placeholder="search"
                className="w-full max-w-[420px] rounded-lg border border-[var(--border-strong)] bg-[var(--surface-2)] px-4 py-2 text-sm text-[var(--text-strong)] outline-none transition focus:border-[var(--accent)]"
              />
            </div>
          </div>
        ) : (
          <div className="flex-1" />
        )}

        <nav className="flex items-center gap-2 shrink-0">
          <AuthButton />
          <LeaderboardButton />
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
