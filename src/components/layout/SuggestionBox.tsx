"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import type { User } from "@supabase/supabase-js";

export function SuggestionBox() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [value, setValue] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [user, setUser] = useState<User | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data: { user } }) => setUser(user));

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => setUser(session?.user ?? null)
    );

    return () => subscription.unsubscribe();
  }, []);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const maxHeight = 160;
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [value, adjustHeight]);

  useEffect(() => {
    if (isOpen && textareaRef.current) {
      textareaRef.current.focus();
    }
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
        }),
      });
      if (!res.ok) throw new Error("Failed to submit");
      setStatus("sent");
      setValue("");
      setTimeout(() => {
        setStatus("idle");
        setIsOpen(false);
      }, 2000);
    } catch {
      setStatus("error");
      setTimeout(() => setStatus("idle"), 3000);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setValue("");
    setStatus("idle");
  };

  const hasContent = value.trim().length > 0;

  return (
    <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
      <div className="flex items-center gap-2">
        <div className="rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 shadow-lg backdrop-blur-lg transition-all duration-300 ease-in-out"
          style={{ width: isOpen ? "min(340px, calc(100vw - 5rem))" : "auto" }}
        >
          {!isOpen ? (
            user ? (
              <button
                onClick={() => setIsOpen(true)}
                className="w-full whitespace-nowrap px-5 py-2.5 text-sm font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-strong)]"
              >
                Suggest an improvement
              </button>
            ) : (
              <a
                href={`/login?redirectTo=${encodeURIComponent(pathname)}`}
                className="block w-full whitespace-nowrap px-5 py-2.5 text-sm font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-strong)]"
              >
                Sign in to suggest
              </a>
            )
          ) : status === "sent" ? (
            <span className="block px-5 py-2.5 text-sm text-[var(--text-muted)]">
              Thank you for your suggestion
            </span>
          ) : (
            <div className="flex items-center gap-2 px-4 py-2">
              <textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    handleSubmit();
                  }
                  if (e.key === "Escape") {
                    handleClose();
                  }
                }}
                placeholder="What could be better?"
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-[var(--text-strong)] placeholder-[var(--text-soft)] outline-none"
                style={{ minHeight: "24px", maxHeight: "160px" }}
              />
              <button
                onClick={handleClose}
                className="shrink-0 rounded-md p-1 text-[var(--text-soft)] transition-colors hover:text-[var(--text-strong)]"
                aria-label="Close"
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
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
            className="shrink-0 rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 px-4 py-2.5 text-sm text-[var(--text-muted)] shadow-lg backdrop-blur-lg transition-colors hover:text-[var(--text-strong)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {status === "sending" ? "..." : "Submit"}
          </button>
        )}
      </div>
    </div>
  );
}
