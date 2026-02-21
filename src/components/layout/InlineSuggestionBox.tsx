"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import type { User } from "@supabase/supabase-js";

export function InlineSuggestionBox() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [value, setValue] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [user, setUser] = useState<User | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => setUser(user));
  }, []);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
  }, []);

  useEffect(() => { adjustHeight(); }, [value, adjustHeight]);

  useEffect(() => {
    if (isOpen && textareaRef.current) textareaRef.current.focus();
  }, [isOpen]);

  const handleSubmit = async () => {
    const trimmed = value.trim();
    if (!trimmed || status === "sending") return;
    setStatus("sending");
    try {
      const res = await fetch("/api/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          suggestion: trimmed,
          page: pathname,
          timestamp: new Date().toISOString(),
          context: "simulation-fullscreen",
        }),
      });
      if (!res.ok) throw new Error("Failed to submit");
      setStatus("sent");
      setValue("");
      setTimeout(() => { setStatus("idle"); setIsOpen(false); }, 2000);
    } catch {
      setStatus("error");
      setTimeout(() => setStatus("idle"), 3000);
    }
  };

  const handleClose = () => { setIsOpen(false); setValue(""); setStatus("idle"); };
  const hasContent = value.trim().length > 0;

  if (!user) return null;

  return (
    <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-[15] pointer-events-auto">
      <div className="flex items-center gap-2">
        <div
          className="rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 shadow-lg backdrop-blur-lg transition-all duration-300"
          style={{ width: isOpen ? "min(320px, calc(100vw - 6rem))" : "auto" }}
        >
          {!isOpen ? (
            <button
              onClick={() => setIsOpen(true)}
              className="whitespace-nowrap px-4 py-2 text-xs font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-strong)]"
            >
              Suggest
            </button>
          ) : status === "sent" ? (
            <span className="block px-4 py-2 text-xs text-[var(--text-muted)]">Thanks!</span>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5">
              <textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
                  if (e.key === "Escape") handleClose();
                }}
                placeholder="What could be better?"
                rows={1}
                className="flex-1 resize-none bg-transparent text-xs text-[var(--text-strong)] placeholder-[var(--text-soft)] outline-none"
                style={{ minHeight: "20px", maxHeight: "120px" }}
              />
              <button
                onClick={handleClose}
                className="shrink-0 rounded-md p-1 text-[var(--text-soft)] transition-colors hover:text-[var(--text-strong)]"
                aria-label="Close"
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M4 4l8 8M12 4l-8 8" />
                </svg>
              </button>
            </div>
          )}
        </div>

        {isOpen && hasContent && status !== "sent" && (
          <button
            onClick={handleSubmit}
            disabled={status === "sending"}
            className="shrink-0 rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 px-3 py-2 text-xs text-[var(--text-muted)] shadow-lg backdrop-blur-lg transition-colors hover:text-[var(--text-strong)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {status === "sending" ? "..." : "Submit"}
          </button>
        )}
      </div>
    </div>
  );
}
