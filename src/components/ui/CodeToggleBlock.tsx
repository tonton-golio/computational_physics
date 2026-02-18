'use client';

import { useState } from 'react';

interface CodeToggleBlockProps {
  pythonCode: string;
  pseudoCode: string;
}

export function CodeToggleBlock({ pythonCode, pseudoCode }: CodeToggleBlockProps) {
  const [mode, setMode] = useState<'python' | 'pseudocode'>('python');

  return (
    <div className="relative my-4 rounded-lg overflow-hidden bg-[#0a0a15] border border-[var(--border-strong)]">
      {/* Toggle tabs */}
      <div className="flex items-center justify-end gap-1 px-3 py-1.5 bg-[#0d0d1a] border-b border-[var(--border-strong)]">
        <button
          onClick={() => setMode('python')}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            mode === 'python'
              ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Python
        </button>
        <button
          onClick={() => setMode('pseudocode')}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            mode === 'pseudocode'
              ? 'bg-purple-600/30 text-purple-300 border border-purple-500/40'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Pseudocode
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-sm text-gray-100">
        <code>{mode === 'python' ? pythonCode : pseudoCode}</code>
      </pre>
    </div>
  );
}
