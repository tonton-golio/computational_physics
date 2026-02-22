"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

/** Check whether the native Fullscreen API is available (false on iOS iPhone). */
function isFullscreenApiAvailable(): boolean {
  if (typeof document === "undefined") return false;
  return !!(
    document.fullscreenEnabled ||
    (document as unknown as { webkitFullscreenEnabled?: boolean })
      .webkitFullscreenEnabled
  );
}

const FAKE_FS_CLASS = "sim-fake-fullscreen";

export function useFullscreen(ref: RefObject<HTMLElement | null>) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const savedOverflowRef = useRef<string>("");

  // ── Native Fullscreen API listeners ──
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

  // ── Cleanup CSS fallback on unmount ──
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el && el.classList.contains(FAKE_FS_CLASS)) {
        el.style.removeProperty("position");
        el.style.removeProperty("inset");
        el.style.removeProperty("z-index");
        el.style.removeProperty("width");
        el.style.removeProperty("height");
        el.classList.remove(FAKE_FS_CLASS);
        document.body.style.overflow = savedOverflowRef.current;
      }
    };
  }, [ref]);

  const toggle = useCallback(() => {
    const el = ref.current;
    if (!el) return;

    // ── CSS fallback path (iOS iPhone) ──
    if (!isFullscreenApiAvailable()) {
      if (el.classList.contains(FAKE_FS_CLASS)) {
        // Exit fake fullscreen
        el.style.removeProperty("position");
        el.style.removeProperty("inset");
        el.style.removeProperty("z-index");
        el.style.removeProperty("width");
        el.style.removeProperty("height");
        el.classList.remove(FAKE_FS_CLASS);
        document.body.style.overflow = savedOverflowRef.current;
        setIsFullscreen(false);
      } else {
        // Enter fake fullscreen
        savedOverflowRef.current = document.body.style.overflow;
        el.style.position = "fixed";
        el.style.inset = "0";
        el.style.zIndex = "9999";
        el.style.width = "100vw";
        el.style.height = "100dvh";
        el.classList.add(FAKE_FS_CLASS);
        document.body.style.overflow = "hidden";
        setIsFullscreen(true);
      }
      return;
    }

    // ── Native Fullscreen API path ──
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
