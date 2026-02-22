"use client";

import { useEffect } from "react";

interface ModalProps extends React.HTMLAttributes<HTMLDivElement> {
  onClose: () => void;
  children: React.ReactNode;
}

export function Modal({ onClose, children, className, ...rest }: ModalProps) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  return (
    <div
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
