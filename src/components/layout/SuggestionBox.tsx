"use client";

import { usePathname } from "next/navigation";
import { useCurrentUser } from "@/lib/use-current-user";
import { useSuggestionForm } from "@/lib/use-suggestion-form";
import { CloseIcon } from "@/components/ui/close-icon";

type SuggestionBoxVariant = "default" | "inline";

interface SuggestionBoxProps {
  variant?: SuggestionBoxVariant;
}

export function SuggestionBox({ variant = "default" }: SuggestionBoxProps) {
  const inline = variant === "inline";
  const pathname = usePathname();
  const user = useCurrentUser();
  const {
    isOpen, setIsOpen, value, setValue, status,
    textareaRef, handleSubmit, handleClose, hasContent,
  } = useSuggestionForm({
    context: inline ? "simulation-fullscreen" : undefined,
    maxHeight: inline ? 120 : 160,
  });

  if (inline && !user) return null;

  return (
    <div className={inline
      ? "absolute bottom-3 left-1/2 -translate-x-1/2 z-[15] pointer-events-auto"
      : "fixed bottom-6 left-1/2 z-50 -translate-x-1/2"
    }>
      <div className="flex items-center gap-2">
        <div
          className="rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 shadow-lg backdrop-blur-lg transition-all duration-300 ease-in-out"
          style={{ width: isOpen ? `min(${inline ? 320 : 340}px, calc(100vw - ${inline ? "6rem" : "5rem"}))` : "auto" }}
        >
          {!isOpen ? (
            !inline && !user ? (
              <a
                href={`/login?redirectTo=${encodeURIComponent(pathname)}`}
                className={`block w-full whitespace-nowrap ${inline ? "px-4 py-2 text-xs" : "px-5 py-2.5 text-sm"} font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-strong)]`}
              >
                Sign in to suggest
              </a>
            ) : (
              <button
                onClick={() => setIsOpen(true)}
                className={`${inline ? "" : "w-full "}whitespace-nowrap ${inline ? "px-4 py-2 text-xs" : "px-5 py-2.5 text-sm"} font-medium text-[var(--text-muted)] transition-colors hover:text-[var(--text-strong)]`}
              >
                {inline ? "Suggest" : "Suggest an improvement"}
              </button>
            )
          ) : status === "sent" ? (
            <span className={`block ${inline ? "px-4 py-2 text-xs" : "px-5 py-2.5 text-sm"} text-[var(--text-muted)]`}>
              {inline ? "Thanks!" : "Thank you for your suggestion"}
            </span>
          ) : (
            <div className={`flex items-center gap-2 ${inline ? "px-3 py-1.5" : "px-4 py-2"}`}>
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
                className={`flex-1 resize-none bg-transparent ${inline ? "text-xs" : "text-sm"} text-[var(--text-strong)] placeholder-[var(--text-soft)] outline-none`}
                style={{ minHeight: inline ? "20px" : "24px", maxHeight: inline ? "120px" : "160px" }}
              />
              <button
                onClick={handleClose}
                className="shrink-0 rounded-md p-1 text-[var(--text-soft)] transition-colors hover:text-[var(--text-strong)]"
                aria-label="Close"
              >
                <CloseIcon size={inline ? 12 : 14} />
              </button>
            </div>
          )}
        </div>

        {isOpen && hasContent && status !== "sent" && (
          <button
            onClick={handleSubmit}
            disabled={status === "sending"}
            className={`shrink-0 rounded-full border border-[var(--border-strong)] bg-[var(--surface-1)]/30 ${inline ? "px-3 py-2 text-xs" : "px-4 py-2.5 text-sm"} text-[var(--text-muted)] shadow-lg backdrop-blur-lg transition-colors hover:text-[var(--text-strong)] disabled:cursor-not-allowed disabled:opacity-40`}
          >
            {status === "sending" ? "..." : "Submit"}
          </button>
        )}
      </div>
    </div>
  );
}
