import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center px-4">
      <div className="w-full max-w-xl rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 text-[var(--text-strong)]">
        <h2 className="text-lg font-semibold">Page not found</h2>
        <p className="mt-2 text-sm text-[var(--text-muted)]">The route you requested does not exist.</p>
        <Link href="/topics" className="mt-4 inline-block rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500">
          Go to topics
        </Link>
      </div>
    </div>
  );
}
