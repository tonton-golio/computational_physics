"use client";

export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex items-center justify-center bg-[#0a0a14] px-4">
          <div className="w-full max-w-xl rounded-xl border border-red-500/30 bg-[#151525] p-6 text-white">
            <h1 className="text-xl font-semibold text-red-300">Application failed to load</h1>
            <p className="mt-2 text-sm text-gray-300">{error.message || "Unexpected startup failure."}</p>
          </div>
        </div>
      </body>
    </html>
  );
}
