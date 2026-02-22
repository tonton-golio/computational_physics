"use client";

import { useState } from 'react';

interface CodeToggleBlockProps {
  pythonCode: string;
  pseudoCode: string;
}

export function CodeToggleBlock({ pythonCode, pseudoCode }: CodeToggleBlockProps) {
  const [mode, setMode] = useState<'python' | 'pseudocode'>('python');

  return (
    <div className="relative my-4 rounded-lg overflow-hidden bg-[var(--surface-1)] border border-[var(--border-strong)]">
      {/* Toggle tabs */}
      <div className="flex items-center justify-end gap-1 px-3 py-1.5 bg-[var(--surface-2)] border-b border-[var(--border-strong)]">
        <button
          onClick={() => setMode('python')}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            mode === 'python'
              ? 'bg-[color-mix(in_srgb,var(--accent)_20%,transparent)] text-[var(--accent)] border border-[var(--accent)]/40'
              : 'text-[var(--text-muted)] hover:text-[var(--text-strong)]'
          }`}
        >
          Python
        </button>
        <button
          onClick={() => setMode('pseudocode')}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            mode === 'pseudocode'
              ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
              : 'text-[var(--text-muted)] hover:text-[var(--text-strong)]'
          }`}
        >
          Pseudocode
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm text-[var(--foreground)]">
        <code>{mode === 'python' ? pythonCode : pseudoCode}</code>
      </pre>
    </div>
  );
}
