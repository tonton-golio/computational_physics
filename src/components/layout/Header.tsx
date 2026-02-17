 "use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ThemeToggle } from "@/components/layout/ThemeToggle";

export function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const isTopics = pathname === "/topics";
  const view = searchParams.get("view") === "box-view" ? "box-view" : "point-cloud";
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
    <header className="sticky top-0 z-50 border-b border-[var(--border-strong)] bg-[var(--background)] backdrop-blur-sm">
      <div className="flex h-16 w-full items-center gap-3 px-6">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-xl font-semibold text-[var(--text-strong)]">koala brain</span>
        </Link>

        {isTopics ? (
          <div className="flex min-w-0 flex-1 justify-center">
            <div className="flex w-full max-w-[720px] items-center gap-3">
              <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={() => updateTopicsParam("view", "box-view")}
                className={`rounded-md border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition ${
                  view === "box-view"
                    ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--accent)]"
                    : "border-[var(--border-strong)] bg-[var(--surface-1)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                }`}
              >
                box-view
              </button>
              <button
                type="button"
                onClick={() => updateTopicsParam("view", "point-cloud")}
                className={`rounded-md border px-3 py-1 text-xs font-semibold uppercase tracking-wide transition ${
                  view === "point-cloud"
                    ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--accent)]"
                    : "border-[var(--border-strong)] bg-[var(--surface-1)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                }`}
              >
                point-cloud
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

        <nav className="flex items-center shrink-0">
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
