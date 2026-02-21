import { cn } from "@/lib/utils";

interface FullscreenButtonProps {
  isFullscreen: boolean;
  onClick: () => void;
  className?: string;
}

export function FullscreenButton({
  isFullscreen,
  onClick,
  className,
}: FullscreenButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
      className={cn(
        "flex h-8 w-8 items-center justify-center rounded-md",
        "bg-[var(--surface-2)]/70 backdrop-blur-sm",
        "border border-[var(--border-strong)]",
        "text-[var(--text-soft)] hover:text-[var(--text-strong)]",
        "hover:bg-[var(--surface-3)]/80",
        "transition-colors cursor-pointer",
        className,
      )}
    >
      {isFullscreen ? (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="4 14 10 14 10 20" />
          <polyline points="20 10 14 10 14 4" />
          <line x1="14" y1="10" x2="21" y2="3" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      ) : (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" y1="3" x2="14" y2="10" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      )}
    </button>
  );
}
