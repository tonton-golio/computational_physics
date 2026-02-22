"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";

interface UseSuggestionFormOptions {
  context?: string;
  maxHeight?: number;
}

export function useSuggestionForm({ context, maxHeight = 160 }: UseSuggestionFormOptions = {}) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);
  const [value, setValue] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
  }, [maxHeight]);

  useEffect(() => { adjustHeight(); }, [value, adjustHeight]);

  useEffect(() => {
    if (isOpen && textareaRef.current) textareaRef.current.focus();
  }, [isOpen]);

  const handleSubmit = async () => {
    const trimmed = value.trim();
    if (!trimmed || status === "sending") return;

    const resolvedContext = context
      ?? (typeof document !== "undefined" && document.querySelector("[data-figure-lightbox]")
        ? "figure-fullscreen"
        : undefined);

    setStatus("sending");
    try {
      const res = await fetch("/api/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          suggestion: trimmed,
          page: pathname,
          timestamp: new Date().toISOString(),
          ...(resolvedContext && { context: resolvedContext }),
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

  const handleClose = () => {
    setIsOpen(false);
    setValue("");
    setStatus("idle");
  };

  return {
    isOpen,
    setIsOpen,
    value,
    setValue,
    status,
    textareaRef,
    handleSubmit,
    handleClose,
    hasContent: value.trim().length > 0,
  };
}
