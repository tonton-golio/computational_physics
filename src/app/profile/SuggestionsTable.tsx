"use client";

import { useState } from "react";

interface Suggestion {
  id: string;
  suggestion: string;
  page: string;
  status: string;
  created_at: string;
}

export function SuggestionsTable({ suggestions }: { suggestions: Suggestion[] }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-[var(--border-strong)]">
      <table className="w-full text-left text-sm">
        <thead className="border-b border-[var(--border-strong)] bg-[var(--surface-1)]">
          <tr>
            <th className="px-4 py-2 font-medium text-[var(--text-muted)]">Suggestion</th>
            <th className="px-4 py-2 font-medium text-[var(--text-muted)]">Page</th>
            <th className="px-4 py-2 font-medium text-[var(--text-muted)]">Status</th>
            <th className="px-4 py-2 font-medium text-[var(--text-muted)]">Date</th>
          </tr>
        </thead>
        <tbody>
          {suggestions.map((s) => {
            const expanded = expandedId === s.id;
            return (
              <tr key={s.id} className="border-b border-[var(--border-strong)] last:border-0">
                <td
                  className={`max-w-[240px] cursor-pointer px-4 py-2 text-[var(--text-strong)] ${
                    expanded ? "whitespace-pre-wrap break-words" : "truncate"
                  }`}
                  onClick={() => setExpandedId(expanded ? null : s.id)}
                >
                  {s.suggestion}
                </td>
                <td className="whitespace-nowrap px-4 py-2 text-[var(--text-muted)]">
                  {s.page}
                </td>
                <td className="px-4 py-2">
                  <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                    s.status === "implemented"
                      ? "bg-green-500/10 text-green-400"
                      : s.status === "rejected"
                        ? "bg-red-500/10 text-red-400"
                        : "bg-[var(--surface-2)] text-[var(--text-muted)]"
                  }`}>
                    {s.status}
                  </span>
                </td>
                <td className="whitespace-nowrap px-4 py-2 text-[var(--text-muted)]">
                  {new Date(s.created_at).toLocaleDateString()}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
