import { createClient } from "@/infra/supabase/server";

export const metadata = {
  title: "Leaderboard | Koala Brain",
};

type LeaderboardEntry = {
  user_id: string;
  display_name: string;
  avatar_url: string | null;
  implemented_count: number;
};

export default async function LeaderboardPage() {
  const supabase = await createClient();

  const { data } = await supabase.rpc("get_leaderboard");
  const entries: LeaderboardEntry[] = data ?? [];

  return (
    <div className="mx-auto max-w-2xl px-4 py-12">
      <h1 className="text-2xl font-semibold text-[var(--text-strong)]">
        Leaderboard
      </h1>
      <p className="mt-1 text-sm text-[var(--text-muted)]">
        Users ranked by implemented suggestions
      </p>

      {entries.length === 0 ? (
        <p className="mt-8 text-sm text-[var(--text-muted)]">
          No implemented suggestions yet. Be the first!
        </p>
      ) : (
        <ol className="mt-8 space-y-2">
          {entries.map((entry, index) => (
            <li
              key={entry.user_id}
              className="flex items-center gap-4 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] px-5 py-3"
            >
              <span className="w-8 text-center text-sm font-semibold text-[var(--text-muted)]">
                {index + 1}
              </span>

              {entry.avatar_url ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={entry.avatar_url}
                  alt=""
                  className="h-8 w-8 rounded-full object-cover"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--accent)] text-sm font-bold text-white">
                  {entry.display_name?.[0]?.toUpperCase() ?? "?"}
                </span>
              )}

              <span className="min-w-0 flex-1 truncate text-sm font-medium text-[var(--text-strong)]">
                {entry.display_name}
              </span>

              <span className="shrink-0 rounded-md bg-[var(--surface-2)] px-3 py-1 text-sm font-semibold text-[var(--accent)]">
                {entry.implemented_count}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
