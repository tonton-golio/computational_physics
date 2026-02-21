"use client";

import { useCallback, useEffect, useState, type RefObject } from "react";

export function useFullscreen(ref: RefObject<HTMLElement | null>) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    function onFullscreenChange() {
      const fsEl =
        document.fullscreenElement ??
        (document as unknown as { webkitFullscreenElement?: Element })
          .webkitFullscreenElement ??
        null;
      setIsFullscreen(fsEl != null && fsEl === ref.current);
    }

    document.addEventListener("fullscreenchange", onFullscreenChange);
    document.addEventListener("webkitfullscreenchange", onFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      document.removeEventListener(
        "webkitfullscreenchange",
        onFullscreenChange,
      );
    };
  }, [ref]);

  const toggle = useCallback(() => {
    const el = ref.current;
    if (!el) return;

    const fsEl =
      document.fullscreenElement ??
      (document as unknown as { webkitFullscreenElement?: Element })
        .webkitFullscreenElement ??
      null;

    if (fsEl) {
      const exitFn =
        document.exitFullscreen?.bind(document) ??
        (document as unknown as { webkitExitFullscreen?: () => Promise<void> })
          .webkitExitFullscreen?.bind(document);
      void exitFn?.();
    } else {
      const requestFn =
        el.requestFullscreen?.bind(el) ??
        (el as unknown as { webkitRequestFullscreen?: () => Promise<void> })
          .webkitRequestFullscreen?.bind(el);
      void requestFn?.();
    }
  }, [ref]);

  return { isFullscreen, toggle } as const;
}
