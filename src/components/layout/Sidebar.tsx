"use client";

import { useState } from "react";
import Link from "next/link";

interface SidebarItem {
  title: string;
  href: string;
  items?: SidebarItem[];
}

interface SidebarProps {
  title: string;
  items: SidebarItem[];
  currentPath?: string;
}

export function Sidebar({ title, items, currentPath }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Mobile toggle */}
      <button
        className="fixed left-4 top-20 z-40 rounded-lg bg-white p-2 shadow-md lg:hidden"
        onClick={() => setIsOpen(!isOpen)}
      >
        <svg
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          {isOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-16 z-30 h-[calc(100vh-4rem)] w-64 transform border-r border-gray-200 bg-white transition-transform lg:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="h-full overflow-y-auto p-4">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-500">
            {title}
          </h2>
          <nav className="space-y-1">
            {items.map((item) => (
              <SidebarLink 
                key={item.href} 
                item={item} 
                currentPath={currentPath} 
              />
            ))}
          </nav>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/20 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}

function SidebarLink({ item, currentPath }: { item: SidebarItem; currentPath?: string }) {
  const isActive = currentPath === item.href;
  
  return (
    <Link
      href={item.href}
      className={`block rounded-lg px-3 py-2 text-sm transition-colors ${
        isActive
          ? "bg-primary-50 font-medium text-primary-700"
          : "text-gray-700 hover:bg-gray-50"
      }`}
    >
      {item.title}
    </Link>
  );
}
