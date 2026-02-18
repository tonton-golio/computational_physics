"use client";

export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex items-center justify-center bg-[var(--background)] px-4">
          <div className="w-full max-w-xl rounded-xl border border-[var(--danger-border)] bg-[var(--surface-1)] p-6 text-[var(--text-strong)]">
            <h1 className="text-xl font-semibold text-red-400">Application failed to load</h1>
            <p className="mt-2 text-sm text-[var(--text-muted)]">{error.message || "Unexpected startup failure."}</p>
          </div>
        </div>
      </body>
    </html>
  );
}
