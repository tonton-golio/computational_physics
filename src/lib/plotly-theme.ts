'use client';

import { useState, useEffect, useCallback } from 'react';
import type Plotly from 'plotly.js';

/** Read a CSS custom-property value from :root. */
export function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

/** Read all theme tokens once and return them. */
function readThemeTokens() {
  return {
    background: cssVar('--surface-2', '#101a2d'),
    surface: cssVar('--surface-1', '#0d1322'),
    textStrong: cssVar('--text-strong', '#f4f7ff'),
    textMuted: cssVar('--text-muted', '#a5b5d0'),
    border: cssVar('--border-strong', '#2c4166'),
  };
}

/** Build a complete Plotly base layout from CSS variables. */
export function buildPlotlyTheme(): Partial<Plotly.Layout> {
  const t = readThemeTokens();
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: t.background,
    font: {
      color: t.textMuted,
      family:
        'var(--font-geist-mono), ui-monospace, SFMono-Regular, Menlo, monospace',
    },
    margin: { t: 52, r: 28, b: 52, l: 62 },
    title: { font: { size: 14, color: t.textStrong } } as unknown as Plotly.DataTitle,
    hoverlabel: {
      bgcolor: t.surface,
      bordercolor: t.border,
      font: { color: t.textStrong },
    },
    legend: {
      bgcolor: t.surface,
      bordercolor: t.border,
      borderwidth: 1,
      font: { color: t.textMuted },
    },
    xaxis: {
      gridcolor: t.border,
      gridwidth: 1,
      zeroline: false,
      zerolinecolor: t.border,
      linecolor: t.border,
      tickfont: { color: t.textMuted },
    },
    yaxis: {
      gridcolor: t.border,
      gridwidth: 1,
      zeroline: false,
      zerolinecolor: t.border,
      linecolor: t.border,
      tickfont: { color: t.textMuted },
    },
  };
}

/**
 * Deep-merge a custom layout on top of the themed base.
 * Handles nested objects like `xaxis.title`, `scene.xaxis`, `colorbar`, etc.
 */
export function mergePlotlyTheme(
  custom: Partial<Plotly.Layout>,
): Partial<Plotly.Layout> {
  const base = buildPlotlyTheme();
  const t = readThemeTokens();

  // Helper: shallow-merge two objects (one level)
  const merge = <T extends Record<string, unknown>>(
    a: T | undefined,
    b: T | undefined,
  ): T => ({ ...(a ?? ({} as T)), ...(b ?? ({} as T)) });

  const mergeAxis = (
    baseAxis: Partial<Plotly.LayoutAxis> | undefined,
    customAxis: Partial<Plotly.LayoutAxis> | undefined,
  ) => {
    if (!customAxis && !baseAxis) return undefined;
    const merged = merge(
      baseAxis as Record<string, unknown>,
      customAxis as Record<string, unknown>,
    ) as Partial<Plotly.LayoutAxis>;

    // Merge nested title/font
    if ((customAxis as Record<string, unknown>)?.title || (baseAxis as Record<string, unknown>)?.title) {
      const baseTitle =
        typeof (baseAxis as Record<string, unknown>)?.title === 'object'
          ? ((baseAxis as Record<string, unknown>).title as Record<string, unknown>)
          : {};
      const customTitle =
        typeof (customAxis as Record<string, unknown>)?.title === 'object'
          ? ((customAxis as Record<string, unknown>).title as Record<string, unknown>)
          : {};
      (merged as Record<string, unknown>).title = {
        ...baseTitle,
        ...customTitle,
        font: merge(
          { color: t.textStrong, ...(baseTitle.font as Record<string, unknown> ?? {}) },
          (customTitle.font ?? {}) as Record<string, unknown>,
        ),
      };
    }

    // Merge nested tickfont
    if (baseAxis?.tickfont || customAxis?.tickfont) {
      merged.tickfont = merge(
        (baseAxis?.tickfont ?? {}) as Record<string, unknown>,
        (customAxis?.tickfont ?? {}) as Record<string, unknown>,
      ) as Partial<Plotly.Font>;
    }

    return merged;
  };

  // Merge scene axes for 3D
  let scene: Partial<Plotly.Scene> | undefined;
  if (custom.scene) {
    scene = {
      ...custom.scene,
      bgcolor: custom.scene.bgcolor ?? t.background,
      xaxis: {
        ...(custom.scene.xaxis ?? {}),
        gridcolor: custom.scene.xaxis?.gridcolor ?? t.border,
        color: (custom.scene.xaxis as Record<string, unknown>)?.color as string ?? t.textMuted,
      },
      yaxis: {
        ...(custom.scene.yaxis ?? {}),
        gridcolor: custom.scene.yaxis?.gridcolor ?? t.border,
        color: (custom.scene.yaxis as Record<string, unknown>)?.color as string ?? t.textMuted,
      },
      zaxis: {
        ...(custom.scene.zaxis ?? {}),
        gridcolor: custom.scene.zaxis?.gridcolor ?? t.border,
        color: (custom.scene.zaxis as Record<string, unknown>)?.color as string ?? t.textMuted,
      },
    };
  }

  const result: Partial<Plotly.Layout> = {
    ...base,
    ...custom,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: custom.plot_bgcolor ?? t.background,
    font: merge(
      base.font as Record<string, unknown>,
      custom.font as Record<string, unknown>,
    ) as Partial<Plotly.Font>,
    hoverlabel: merge(
      base.hoverlabel as Record<string, unknown>,
      custom.hoverlabel as Record<string, unknown>,
    ) as Partial<Plotly.HoverLabel>,
    legend: merge(
      base.legend as Record<string, unknown>,
      custom.legend as Record<string, unknown>,
    ) as Partial<Plotly.Legend>,
    xaxis: mergeAxis(
      base.xaxis as Partial<Plotly.LayoutAxis>,
      custom.xaxis as Partial<Plotly.LayoutAxis>,
    ),
    yaxis: mergeAxis(
      base.yaxis as Partial<Plotly.LayoutAxis>,
      custom.yaxis as Partial<Plotly.LayoutAxis>,
    ),
  };

  if (scene) result.scene = scene;

  // Handle title
  if (custom.title) {
    if (typeof custom.title === 'string') {
      result.title = { text: custom.title, font: { color: t.textStrong } } as unknown as Plotly.DataTitle;
    } else {
      result.title = {
        ...(custom.title as Plotly.DataTitle),
        font: merge(
          { color: t.textStrong },
          ((custom.title as Plotly.DataTitle).font ?? {}) as Record<string, unknown>,
        ),
      } as Plotly.DataTitle;
    }
  }

  return result;
}

/**
 * React hook that re-builds the Plotly theme whenever `data-theme` changes.
 * Returns `{ baseLayout, mergeLayout }` that auto-update on theme toggle.
 */
export function usePlotlyTheme() {
  const [, setVersion] = useState(0);

  useEffect(() => {
    const observer = new MutationObserver(() => setVersion((v) => v + 1));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    return () => observer.disconnect();
  }, []);

  const baseLayout = buildPlotlyTheme();
  const mergeLayout = useCallback(
    (custom: Partial<Plotly.Layout>) => mergePlotlyTheme(custom),
    [],
  );

  return { baseLayout, mergeLayout };
}
