"use client";

import { useEffect, useState } from "react";
import { type Theme, resolveInitialTheme, applyTheme } from "@/lib/use-theme";
import { ICON_BUTTON_CLASS } from "@/components/ui/icon-button";
import { SunIcon, MoonIcon } from "@/components/ui/icons";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    setTheme(resolveInitialTheme());
  }, []);

  useEffect(() => {
    if (theme) applyTheme(theme);
  }, [theme]);

  if (!theme) return null;

  return (
    <button
      type="button"
      onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
      className={ICON_BUTTON_CLASS}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}
