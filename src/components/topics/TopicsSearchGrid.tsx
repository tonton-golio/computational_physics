"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import type { WheelEvent as ReactWheelEvent } from "react";
import { topicHref } from "@/lib/topic-navigation";
import { useTheme } from "@/lib/use-theme";
import { ExportPdfButton } from "@/components/content/ExportPdfButton";

export interface TopicsSearchEntry {
  routeSlug: string;
  topicId: string;
  meta: {
    title: string;
    description?: string;
  };
  lessons: Array<{
    slug: string;
    title: string;
    summary?: string;
    searchableText: string;
  }>;
}

interface TopicsSearchGridProps {
  entries: TopicsSearchEntry[];
}

interface PointCloudPoint {
  topicId: string;
  topicTitle: string;
  routeSlug: string;
  lessonSlug: string;
  lessonTitle: string;
  x: number;
  y: number;
  searchText?: string;
}

interface PointCloudData {
  points: PointCloudPoint[];
}

const TOPIC_COLORS_DARK: Record<string, string> = {
  "quantum-optics": "#b392d6",
  "continuum-mechanics": "#7ba3d4",
  "inverse-problems": "#6db88a",
  "complex-physics": "#d4a373",
  "scientific-computing": "#5fb8c9",
  "online-reinforcement-learning": "#c98aab",
  "advanced-deep-learning": "#c9827e",
  "applied-statistics": "#c4b06a",
  "applied-machine-learning": "#8b8ec4",
  "dynamical-models": "#6aad9e",
};

const TOPIC_COLORS_LIGHT: Record<string, string> = {
  "quantum-optics": "#7b4db0",
  "continuum-mechanics": "#3d6fa8",
  "inverse-problems": "#2e8854",
  "complex-physics": "#b07838",
  "scientific-computing": "#2890a3",
  "online-reinforcement-learning": "#a8506e",
  "advanced-deep-learning": "#a84f4a",
  "applied-statistics": "#8a7a1e",
  "applied-machine-learning": "#5558a0",
  "dynamical-models": "#3a8672",
};

function topicColor(topicId: string, theme: "light" | "dark"): string {
  const palette = theme === "light" ? TOPIC_COLORS_LIGHT : TOPIC_COLORS_DARK;
  return palette[topicId] ?? (theme === "light" ? "#4a5568" : "#64748b");
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function stableHash(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function shortLabel(value: string, max = 24): string {
  return value.length > max ? `${value.slice(0, max - 1)}...` : value;
}

function pointerDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function TopicsSearchGrid({ entries }: TopicsSearchGridProps) {
  const searchParams = useSearchParams();
  const theme = useTheme();
  const [hasHydrated, setHasHydrated] = useState(false);
  const [pointCloudData, setPointCloudData] = useState<PointCloudData | null>(null);
  const [pointCloudError, setPointCloudError] = useState<string | null>(null);
  const [nearestTopicId, setNearestTopicId] = useState<string | null>(null);
  const nearestTopicRef = useRef<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ point: PointCloudPoint; x: number; y: number } | null>(null);
  const cloudRef = useRef<HTMLDivElement | null>(null);
  const pointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinchStartDistRef = useRef<number | null>(null);
  const pinchStartZoomRef = useRef<number>(1);

  const normalizedQuery = (searchParams.get("q") ?? "").trim().toLowerCase();
  const view = searchParams.get("view") === "points" ? "points" : "boxes";

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  useEffect(() => {
    let active = true;
    async function loadPointCloud() {
      try {
        const response = await fetch("/data/topic-points.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Unable to load points data.");
        }
        const data = (await response.json()) as PointCloudData;
        if (!active) return;
        setPointCloudData(data);
        setPointCloudError(null);
      } catch (error) {
        if (!active) return;
        setPointCloudData(null);
        setPointCloudError(error instanceof Error ? error.message : "Unable to load points data.");
      }
    }
    void loadPointCloud();
    return () => {
      active = false;
    };
  }, []);

  const lessonSummaryMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const entry of entries) {
      for (const lesson of entry.lessons) {
        if (lesson.summary) {
          map.set(`${entry.routeSlug}:${lesson.slug}`, lesson.summary);
        }
      }
    }
    return map;
  }, [entries]);

  const filteredEntries = useMemo(() => {
    if (!normalizedQuery) return entries;

    return entries
      .map((entry) => ({
        ...entry,
        lessons: entry.lessons.filter((lesson) => lesson.searchableText.includes(normalizedQuery)),
      }))
      .filter((entry) => entry.lessons.length > 0);
  }, [entries, normalizedQuery]);

  const lessonMatchCount = useMemo(
    () => filteredEntries.reduce((count, entry) => count + entry.lessons.length, 0),
    [filteredEntries]
  );

  const filteredPoints = useMemo(() => {
    const points = pointCloudData?.points ?? [];
    if (!normalizedQuery) return points;
    return points.filter((point) => {
      const haystack = `${point.lessonTitle}\n${point.topicTitle}\n${point.searchText ?? ""}`.toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  }, [pointCloudData, normalizedQuery]);

  const plottedPoints = useMemo(
    () =>
      filteredPoints.map((point) => {
        const base = stableHash(`${point.topicId}:${point.lessonSlug}`);
        const jx = ((base & 1023) / 1023 - 0.5) * 0.04;
        const jy = (((base >> 10) & 1023) / 1023 - 0.5) * 0.04;
        const worldX = clamp(point.x + jx, 0.01, 0.99);
        const worldY = clamp(point.y + jy, 0.01, 0.99);
        return { ...point, worldX, worldY, hash: base };
      }),
    [filteredPoints]
  );

  const transformedPoints = useMemo(() => {
    return plottedPoints
      .map((point) => {
        const x = (point.worldX - 0.5) * zoom + 0.5 + pan.x;
        const y = (point.worldY - 0.5) * zoom + 0.5 + pan.y;
        return { ...point, tx: x, ty: y };
      })
      .filter((point) => point.tx >= -0.08 && point.tx <= 1.08 && point.ty >= -0.08 && point.ty <= 1.08)
      .sort((a, b) => a.hash - b.hash);
  }, [plottedPoints, zoom, pan.x, pan.y]);

  const topicCentroids = useMemo(() => {
    const groups = new Map<string, { topicTitle: string; routeSlug: string; sumX: number; sumY: number; count: number }>();
    for (const point of transformedPoints) {
      const g = groups.get(point.topicId) ?? { topicTitle: point.topicTitle, routeSlug: point.routeSlug, sumX: 0, sumY: 0, count: 0 };
      g.sumX += point.tx;
      g.sumY += point.ty;
      g.count += 1;
      groups.set(point.topicId, g);
    }
    return Array.from(groups.entries()).map(([topicId, g]) => ({
      topicId,
      topicTitle: g.topicTitle,
      routeSlug: g.routeSlug,
      cx: g.sumX / g.count,
      cy: g.sumY / g.count,
    }));
  }, [transformedPoints]);

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setHoveredPoint(null);
  }

  useEffect(() => {
    if (view !== "points") return;
    resetView();
  }, [view, normalizedQuery]);

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const container = cloudRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const cx = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const cy = clamp((event.clientY - rect.top) / rect.height, 0, 1);
    const nextZoom = clamp(zoom * (event.deltaY > 0 ? 0.9 : 1.1), 0.8, 4);

    const worldX = (cx - 0.5 - pan.x) / zoom + 0.5;
    const worldY = (cy - 0.5 - pan.y) / zoom + 0.5;
    const nextPanX = cx - 0.5 - (worldX - 0.5) * nextZoom;
    const nextPanY = cy - 0.5 - (worldY - 0.5) * nextZoom;

    setZoom(nextZoom);
    setPan({ x: clamp(nextPanX, -1.5, 1.5), y: clamp(nextPanY, -1.5, 1.5) });
  }

  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    (event.currentTarget as HTMLDivElement).setPointerCapture(event.pointerId);

    if (pointersRef.current.size === 1) {
      setDragStart({ x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y });
    } else if (pointersRef.current.size === 2) {
      // Start pinch
      setDragStart(null);
      const pts = Array.from(pointersRef.current.values());
      pinchStartDistRef.current = pointerDistance(pts[0], pts[1]);
      pinchStartZoomRef.current = zoom;
    }
  }, [pan.x, pan.y, zoom]);

  const handlePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const container = cloudRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();

    if (pointersRef.current.size === 2 && pinchStartDistRef.current !== null) {
      // Pinch zoom
      const pts = Array.from(pointersRef.current.values());
      const dist = pointerDistance(pts[0], pts[1]);
      const scale = dist / pinchStartDistRef.current;
      const nextZoom = clamp(pinchStartZoomRef.current * scale, 0.8, 4);

      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;
      const cx = clamp((midX - rect.left) / rect.width, 0, 1);
      const cy = clamp((midY - rect.top) / rect.height, 0, 1);

      const worldX = (cx - 0.5 - pan.x) / zoom + 0.5;
      const worldY = (cy - 0.5 - pan.y) / zoom + 0.5;
      const nextPanX = cx - 0.5 - (worldX - 0.5) * nextZoom;
      const nextPanY = cy - 0.5 - (worldY - 0.5) * nextZoom;

      setZoom(nextZoom);
      setPan({ x: clamp(nextPanX, -1.5, 1.5), y: clamp(nextPanY, -1.5, 1.5) });
      return;
    }

    if (dragStart && pointersRef.current.size === 1) {
      const dx = (event.clientX - dragStart.x) / rect.width;
      const dy = (event.clientY - dragStart.y) / rect.height;
      setPan({ x: clamp(dragStart.panX + dx, -1.5, 1.5), y: clamp(dragStart.panY + dy, -1.5, 1.5) });
    }

    // Nearest topic tracking (for hover highlighting)
    if (pointersRef.current.size <= 1) {
      const mx = (event.clientX - rect.left) / rect.width;
      const my = (event.clientY - rect.top) / rect.height;
      let minDist = Infinity;
      let nearest: string | null = null;
      for (const p of transformedPoints) {
        const dist = Math.hypot(p.tx - mx, p.ty - my);
        if (dist < minDist) {
          minDist = dist;
          nearest = p.topicId;
        }
      }
      const next = minDist < 0.1 ? nearest : null;
      if (next !== nearestTopicRef.current) {
        nearestTopicRef.current = next;
        setNearestTopicId(next);
      }
    }
  }, [dragStart, pan.x, pan.y, zoom, transformedPoints]);

  const handlePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) {
      pinchStartDistRef.current = null;
    }
    if (pointersRef.current.size === 0) {
      setDragStart(null);
    }
  }, []);

  const handlePointerCancel = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) {
      pinchStartDistRef.current = null;
    }
    if (pointersRef.current.size === 0) {
      setDragStart(null);
      setHoveredPoint(null);
      nearestTopicRef.current = null;
      setNearestTopicId(null);
    }
  }, []);

  const handlePointerLeave = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) {
      pinchStartDistRef.current = null;
    }
    if (pointersRef.current.size === 0) {
      setDragStart(null);
      setHoveredPoint(null);
      nearestTopicRef.current = null;
      setNearestTopicId(null);
    }
  }, []);

  function showTooltip(event: React.MouseEvent<HTMLAnchorElement>, point: PointCloudPoint) {
    const container = cloudRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setHoveredPoint({
      point,
      x: clamp(event.clientX - rect.left + 10, 8, rect.width - 180),
      y: clamp(event.clientY - rect.top - 8, 8, rect.height - 48),
    });
  }

  if (!hasHydrated) {
    return (
      <div className="mt-8 rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 text-center text-sm text-[var(--text-muted)]">
        Loading topic search...
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="px-3 sm:px-6 pt-2">
        <p className="text-xs text-[var(--text-muted)]">
          {normalizedQuery ? `${view === "points" ? filteredPoints.length : lessonMatchCount} matching subtopics` : ""}
        </p>
      </div>

      {view === "points" ? (
        <div className="relative mx-3 sm:mx-6 mt-3 min-h-0 flex-1 overflow-hidden">
          <div
            ref={cloudRef}
            onWheel={handleWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerCancel}
            onPointerLeave={handlePointerLeave}
            className={`relative h-full w-full overflow-hidden ${
              dragStart ? "cursor-grabbing" : "cursor-grab"
            }`}
            style={{ touchAction: "none" }}
          >
            {/* radial gradient is now applied globally on body in dark mode */}
            {pointCloudError ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                {pointCloudError}
              </div>
            ) : !pointCloudData || pointCloudData.points.length === 0 ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                Points data is empty. Run `npm run build:points` first.
              </div>
            ) : filteredPoints.length === 0 ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                No matching subtopics found.
              </div>
            ) : transformedPoints.length === 0 ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                <p>Points are outside the current viewport.</p>
                <button
                  type="button"
                  onClick={resetView}
                  className="mt-2 rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-2 py-1 text-[10px] uppercase tracking-wide text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
                >
                  reset view
                </button>
              </div>
            ) : (
              <>
                <div className="pointer-events-none absolute right-6 top-6 z-20 rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-2 py-1 text-[10px] text-[var(--text-muted)]">
                  {transformedPoints.length}/{filteredPoints.length} points - zoom {zoom.toFixed(2)}x
                </div>
                <button
                  type="button"
                  onClick={resetView}
                  className="absolute left-6 top-6 z-20 rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-2 py-1 text-[10px] text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
                >
                  reset view
                </button>
                {topicCentroids.map((centroid) => (
                  <Link
                    key={`label-${centroid.topicId}`}
                    href={topicHref(centroid.routeSlug)}
                    className="absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-md px-2.5 py-1 text-[11px] font-medium no-underline hover:brightness-125"
                    style={{
                      left: `${centroid.cx * 100}%`,
                      top: `${centroid.cy * 100}%`,
                      backgroundColor: `${topicColor(centroid.topicId, theme)}22`,
                      color: topicColor(centroid.topicId, theme),
                      border: `1px solid ${topicColor(centroid.topicId, theme)}44`,
                      zIndex: nearestTopicId === centroid.topicId ? 1 : 10,
                      opacity: nearestTopicId && nearestTopicId !== centroid.topicId ? 0.3 : 1,
                      transition: "opacity 0.15s ease, z-index 0s",
                    }}
                  >
                    {centroid.topicTitle}
                  </Link>
                ))}
                {transformedPoints.map((point) => {
                  const isHighlighted = nearestTopicId === point.topicId;
                  const isDimmed = nearestTopicId !== null && !isHighlighted;
                  return (
                    <Link
                      key={`${point.topicId}:${point.lessonSlug}`}
                      href={topicHref(point.routeSlug, point.lessonSlug)}
                      title={`${point.topicTitle} - ${point.lessonTitle}`}
                      onMouseEnter={(event) => showTooltip(event, point)}
                      onMouseMove={(event) => showTooltip(event, point)}
                      onMouseLeave={() => setHoveredPoint(null)}
                      className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/20 focus-visible:scale-110"
                      style={{
                        left: `${point.tx * 100}%`,
                        top: `${point.ty * 100}%`,
                        backgroundColor: topicColor(point.topicId, theme),
                        width: isHighlighted ? "13px" : "10px",
                        height: isHighlighted ? "13px" : "10px",
                        opacity: isDimmed ? 0.25 : 1,
                        zIndex: isHighlighted ? 15 : 5,
                        boxShadow: isHighlighted
                          ? `0 0 10px ${topicColor(point.topicId, theme)}, 0 0 4px ${topicColor(point.topicId, theme)}`
                          : "0 1px 8px rgba(0,0,0,0.25)",
                        transition: "width 0.15s ease, height 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease",
                      }}
                      aria-label={`${point.topicTitle}: ${point.lessonTitle}`}
                    />
                  );
                })}
                {hoveredPoint ? (
                  <div
                    className="pointer-events-none absolute z-30 max-w-[280px] rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-2 py-1.5 text-[11px] shadow-md"
                    style={{ left: hoveredPoint.x, top: hoveredPoint.y, transform: "translateY(-100%)" }}
                  >
                    <p className="truncate text-[var(--text-strong)]">{shortLabel(hoveredPoint.point.lessonTitle, 42)}</p>
                    <p className="truncate text-[10px] text-[var(--text-muted)]">{hoveredPoint.point.topicTitle}</p>
                    {lessonSummaryMap.get(`${hoveredPoint.point.routeSlug}:${hoveredPoint.point.lessonSlug}`) && (
                      <p className="mt-1 text-[10px] leading-tight text-[var(--text-soft)]">
                        {lessonSummaryMap.get(`${hoveredPoint.point.routeSlug}:${hoveredPoint.point.lessonSlug}`)}
                      </p>
                    )}
                  </div>
                ) : null}
              </>
            )}
          </div>
        </div>
      ) : filteredEntries.length === 0 ? (
        <div className="mx-3 sm:mx-6 mt-4 rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 text-center text-sm text-[var(--text-muted)]">
          No matching subtopics found.
        </div>
      ) : (
        <div className="mx-3 sm:mx-6 mt-4 min-h-0 flex-1 overflow-auto pr-1">
          <div className="grid gap-4 lg:grid-cols-2">
            {filteredEntries.map((entry) => (
              <div key={entry.routeSlug} className="rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <Link
                      href={topicHref(entry.routeSlug)}
                      className="inline-flex rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-1 text-xs font-semibold uppercase tracking-wide text-[var(--accent)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]"
                    >
                      {entry.meta.title}
                    </Link>
                    <p className="mt-3 text-sm text-[var(--text-muted)]">{entry.meta.description}</p>
                  </div>
                  <ExportPdfButton topicId={entry.topicId} topicTitle={entry.meta.title} variant="compact" />
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {entry.lessons.map((lesson) => (
                    <Link
                      key={lesson.slug}
                      href={topicHref(entry.routeSlug, lesson.slug)}
                      title={lesson.summary}
                      className="rounded-full border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-1 text-xs text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:bg-[var(--surface-3)] hover:text-[var(--text-strong)]"
                    >
                      {lesson.title}
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
