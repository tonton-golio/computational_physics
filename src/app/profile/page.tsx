import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { ProfileActions } from "./ProfileActions";
import { SuggestionsTable } from "./SuggestionsTable";

export const metadata = {
  title: "Profile | Computational Physics",
};

export default async function ProfilePage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { data: suggestions } = await supabase
    .from("suggestions")
    .select("id, suggestion, page, status, created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  const allSuggestions = suggestions ?? [];
  const implemented = allSuggestions.filter((s) => s.status === "implemented").length;

  const avatarUrl = user.user_metadata?.avatar_url as string | undefined;
  const displayName = user.user_metadata?.full_name ?? user.user_metadata?.name ?? null;
  const email = user.email ?? "";
  const initial = email[0]?.toUpperCase() ?? "?";

  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      {/* User info */}
      <div className="flex items-center gap-4">
        {avatarUrl ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={avatarUrl}
            alt=""
            className="h-14 w-14 rounded-full object-cover"
            referrerPolicy="no-referrer"
          />
        ) : (
          <span className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--accent)] text-xl font-bold text-white">
            {initial}
          </span>
        )}
        <div className="min-w-0">
          {displayName && (
            <p className="truncate text-lg font-semibold text-[var(--text-strong)]">
              {displayName}
            </p>
          )}
          <p className="truncate text-sm text-[var(--text-muted)]">{email}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-8 flex gap-6">
        <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] px-5 py-3">
          <p className="text-2xl font-semibold text-[var(--text-strong)]">{allSuggestions.length}</p>
          <p className="text-xs text-[var(--text-muted)]">Suggestions</p>
        </div>
        <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] px-5 py-3">
          <p className="text-2xl font-semibold text-[var(--text-strong)]">{implemented}</p>
          <p className="text-xs text-[var(--text-muted)]">Implemented</p>
        </div>
      </div>

      {/* Suggestions table */}
      <h2 className="mt-10 text-sm font-semibold text-[var(--text-strong)]">
        Your suggestions
      </h2>

      {allSuggestions.length === 0 ? (
        <p className="mt-3 text-sm text-[var(--text-muted)]">
          No suggestions yet. Use the suggestion box at the bottom of any page.
        </p>
      ) : (
        <SuggestionsTable suggestions={allSuggestions} />
      )}

      {/* Actions */}
      <ProfileActions />
    </div>
  );
}
