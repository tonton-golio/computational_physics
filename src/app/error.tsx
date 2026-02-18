"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-[60vh] flex items-center justify-center px-4">
      <div className="w-full max-w-xl rounded-xl border border-[var(--danger-border)] bg-[var(--surface-1)] p-6 text-[var(--text-strong)]">
        <h2 className="text-lg font-semibold text-red-400">Something went wrong</h2>
        <p className="mt-2 text-sm text-[var(--text-muted)]">{error.message || "Unexpected application error."}</p>
        <button
          onClick={reset}
          className="mt-4 rounded-lg bg-[var(--accent)] px-4 py-2 text-sm text-white hover:bg-[var(--accent-strong)]"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
