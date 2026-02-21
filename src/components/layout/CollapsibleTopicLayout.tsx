"use client";

import { useState, type ReactNode } from "react";

interface CollapsibleTopicLayoutProps {
  sidebar: ReactNode;
  children: ReactNode;
}

export function CollapsibleTopicLayout({ sidebar, children }: CollapsibleTopicLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setCollapsed(false)}
        className={`fixed left-3 top-[100px] z-50 hidden lg:flex h-7 w-7 items-center justify-center rounded-full border border-[var(--border-strong)] bg-[var(--surface-2)] text-[var(--text-muted)] shadow-md transition-all duration-300 hover:border-[var(--accent)] hover:text-[var(--accent)] ${
          collapsed ? "opacity-100 scale-100" : "opacity-0 scale-75 pointer-events-none"
        }`}
        aria-label="Expand sidebar"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="m9 18 6-6-6-6" />
        </svg>
      </button>

      <div className="sidebar-grid grid w-full" data-collapsed={collapsed}>
        <div className="relative hidden lg:block min-w-0">
          <div className="sticky top-[88px]">
            <div className="relative">
              <button
                type="button"
                onClick={() => setCollapsed(true)}
                className={`absolute -right-3 top-3 z-10 hidden lg:flex h-7 w-7 items-center justify-center rounded-full border border-[var(--border-strong)] bg-[var(--surface-2)] text-[var(--text-muted)] shadow-md transition-all duration-300 hover:border-[var(--accent)] hover:text-[var(--accent)] ${
                  collapsed ? "opacity-0 pointer-events-none" : "opacity-100"
                }`}
                aria-label="Collapse sidebar"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m15 18-6-6 6-6" />
                </svg>
              </button>
              <div className="overflow-clip">
                <div className="w-[320px]">
                  {sidebar}
                </div>
              </div>
            </div>
          </div>
        </div>
        {children}
      </div>
    </>
  );
}
