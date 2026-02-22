"use client";

import { SuggestionBox } from "@/components/layout/SuggestionBox";
import { Modal } from "@/components/ui/modal";
import { XIcon } from "@/components/ui/icons";

interface FigureLightboxProps {
  src: string;
  alt: string;
  isVideo?: boolean;
  caption?: string;
  onClose: () => void;
}

export function FigureLightbox({
  src,
  alt,
  isVideo,
  caption,
  onClose,
}: FigureLightboxProps) {
  return (
    <Modal onClose={onClose} className="bg-black/80" data-figure-lightbox>
      {/* Caption — top-left frosted glass (matches sim-fs-title) */}
      {caption && (
        <div className="pointer-events-none absolute top-12 left-12 z-10 max-w-[min(420px,60vw)] rounded-md border border-white/10 bg-black/40 px-3 py-1.5 text-xs leading-snug text-white/90 backdrop-blur-md">
          {caption}
        </div>
      )}

      <button
        type="button"
        onClick={onClose}
        aria-label="Close lightbox"
        className="absolute top-4 right-4 flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors cursor-pointer"
      >
        <XIcon size={20} />
      </button>

      <div className="flex flex-col items-center gap-3" onClick={(e) => e.stopPropagation()}>
        {isVideo ? (
          <video
            controls
            autoPlay
            className="max-h-[85vh] max-w-[90vw] rounded-lg"
            src={src}
          >
            Your browser does not support embedded video.
          </video>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt={alt}
            className="max-h-[85vh] max-w-[90vw] rounded-lg object-contain"
          />
        )}
      </div>

      {/* Suggestion box — bottom center */}
      <div className="pointer-events-auto absolute bottom-4 left-1/2 -translate-x-1/2 z-10">
        <SuggestionBox variant="inline" />
      </div>
    </Modal>
  );
}
