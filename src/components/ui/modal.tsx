"use client";

import { useEffect, useRef } from "react";

interface ModalProps extends React.HTMLAttributes<HTMLDivElement> {
  onClose: () => void;
  children: React.ReactNode;
}

export function Modal({ onClose, children, className, ...rest }: ModalProps) {
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;

    const getFocusable = () => {
      const overlay = overlayRef.current;
      if (!overlay) return [] as HTMLElement[];
      return Array.from(
        overlay.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => el.offsetParent !== null || el === overlay);
    };

    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab") {
        const focusable = getFocusable();
        const overlay = overlayRef.current;
        if (focusable.length === 0) {
          e.preventDefault();
          overlay?.focus();
          return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = document.activeElement;
        if (e.shiftKey) {
          if (active === first || active === overlay) {
            e.preventDefault();
            last.focus();
          }
        } else if (active === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKey);
    document.body.style.overflow = "hidden";
    overlayRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = "";
      prev?.focus();
    };
  }, [onClose]);

  return (
    <div
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      tabIndex={-1}
      className={`fixed inset-0 z-[9999] flex items-center justify-center backdrop-blur-sm ${className ?? "bg-black/60"}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      {...rest}
    >
      {children}
    </div>
  );
}
