"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/infra/supabase/client";

export function ProfileActions() {
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  async function handleSignOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  async function handleDeleteAccount() {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }

    setDeleting(true);
    try {
      const res = await fetch("/api/auth/delete-account", { method: "DELETE" });
      if (!res.ok) throw new Error("Failed to delete account");

      const supabase = createClient();
      await supabase.auth.signOut();
      router.push("/");
      router.refresh();
    } catch {
      setDeleting(false);
      setConfirmDelete(false);
    }
  }

  return (
    <div className="mt-12 flex items-center gap-3 border-t border-[var(--border-strong)] pt-8">
      <button
        type="button"
        onClick={handleSignOut}
        className="rounded-lg border border-[var(--border-strong)] bg-[var(--surface-2)] px-4 py-2 text-sm text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
      >
        Sign out
      </button>

      <button
        type="button"
        onClick={handleDeleteAccount}
        disabled={deleting}
        className={`rounded-lg border px-4 py-2 text-sm transition ${
          confirmDelete
            ? "border-red-500/50 bg-red-500/10 text-red-400 hover:bg-red-500/20"
            : "border-[var(--border-strong)] bg-[var(--surface-2)] text-[var(--text-muted)] hover:text-red-400"
        } disabled:opacity-50`}
      >
        {deleting
          ? "Deleting..."
          : confirmDelete
            ? "Confirm delete"
            : "Delete account"}
      </button>

      {confirmDelete && !deleting && (
        <button
          type="button"
          onClick={() => setConfirmDelete(false)}
          className="text-xs text-[var(--text-muted)] hover:text-[var(--text-strong)]"
        >
          Cancel
        </button>
      )}
    </div>
  );
}
