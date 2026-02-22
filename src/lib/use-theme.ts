"use client";

import { useState, useEffect } from 'react';

export type Theme = 'light' | 'dark';

export function resolveInitialTheme(): Theme {
  if (typeof window === "undefined") return "dark";

  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "light" || attr === "dark") return attr;

  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") return saved;

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.style.colorScheme = theme;
  localStorage.setItem("theme", theme);
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>('dark');

  useEffect(() => {
    const root = document.documentElement;
    const read = () => (root.getAttribute('data-theme') === 'light' ? 'light' : 'dark') as Theme;
    setTheme(read());

    const observer = new MutationObserver(() => setTheme(read()));
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  return theme;
}

/** Counter that increments on every theme change — use as a useEffect dependency to trigger canvas redraws. */
export function useThemeChangeKey(): number {
  const [key, setKey] = useState(0);

  useEffect(() => {
    const observer = new MutationObserver(() => setKey(k => k + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  return key;
}
