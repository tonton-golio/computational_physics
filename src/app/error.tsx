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
      <div className="w-full max-w-xl rounded-xl border border-red-500/30 bg-[#151525] p-6 text-white">
        <h2 className="text-lg font-semibold text-red-300">Something went wrong</h2>
        <p className="mt-2 text-sm text-gray-300">{error.message || "Unexpected application error."}</p>
        <button
          onClick={reset}
          className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
