"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, WheelEvent as ReactWheelEvent } from "react";
import { topicHref } from "@/lib/topic-navigation";

export interface TopicsSearchEntry {
  routeSlug: string;
  meta: {
    title: string;
    description?: string;
  };
  lessons: Array<{
    slug: string;
    title: string;
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

const TOPIC_COLORS: Record<string, string> = {
  "quantum-optics": "#a855f7",
  "continuum-mechanics": "#3b82f6",
  "inverse-problems": "#22c55e",
  "complex-physics": "#f97316",
  "scientific-computing": "#06b6d4",
  "online-reinforcement-learning": "#ec4899",
  "advanced-deep-learning": "#ef4444",
  "applied-statistics": "#eab308",
  "applied-machine-learning": "#6366f1",
  "dynamical-models": "#14b8a6",
};

function topicColor(topicId: string): string {
  return TOPIC_COLORS[topicId] ?? "#64748b";
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

export function TopicsSearchGrid({ entries }: TopicsSearchGridProps) {
  const searchParams = useSearchParams();
  const [hasHydrated, setHasHydrated] = useState(false);
  const [pointCloudData, setPointCloudData] = useState<PointCloudData | null>(null);
  const [pointCloudError, setPointCloudError] = useState<string | null>(null);
  const [activeTopicId, setActiveTopicId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<{ point: PointCloudPoint; x: number; y: number } | null>(null);
  const cloudRef = useRef<HTMLDivElement | null>(null);

  const normalizedQuery = (searchParams.get("q") ?? "").trim().toLowerCase();
  const view = searchParams.get("view") === "box-view" ? "box-view" : "point-cloud";

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  useEffect(() => {
    let active = true;
    async function loadPointCloud() {
      try {
        const response = await fetch("/data/topic-point-cloud.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Unable to load point-cloud data.");
        }
        const data = (await response.json()) as PointCloudData;
        if (!active) return;
        setPointCloudData(data);
        setPointCloudError(null);
      } catch (error) {
        if (!active) return;
        setPointCloudData(null);
        setPointCloudError(error instanceof Error ? error.message : "Unable to load point-cloud data.");
      }
    }
    void loadPointCloud();
    return () => {
      active = false;
    };
  }, []);

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

  const visiblePoints = useMemo(() => {
    if (!activeTopicId) return filteredPoints;
    return filteredPoints.filter((point) => point.topicId === activeTopicId);
  }, [filteredPoints, activeTopicId]);

  const pointTopicStats = useMemo(() => {
    const totals = new Map<string, { topicTitle: string; total: number; visible: number }>();
    for (const point of pointCloudData?.points ?? []) {
      const current = totals.get(point.topicId) ?? { topicTitle: point.topicTitle, total: 0, visible: 0 };
      current.total += 1;
      totals.set(point.topicId, current);
    }
    for (const point of filteredPoints) {
      const current = totals.get(point.topicId) ?? { topicTitle: point.topicTitle, total: 0, visible: 0 };
      current.visible += 1;
      totals.set(point.topicId, current);
    }
    return Array.from(totals.entries())
      .map(([topicId, value]) => ({ topicId, ...value }))
      .sort((a, b) => a.topicTitle.localeCompare(b.topicTitle));
  }, [pointCloudData, filteredPoints]);

  const plottedPoints = useMemo(
    () =>
      visiblePoints.map((point) => {
        const base = stableHash(`${point.topicId}:${point.lessonSlug}`);
        const jx = ((base & 1023) / 1023 - 0.5) * 0.04;
        const jy = (((base >> 10) & 1023) / 1023 - 0.5) * 0.04;
        const worldX = clamp(point.x + jx, 0.01, 0.99);
        const worldY = clamp(point.y + jy, 0.01, 0.99);
        return { ...point, worldX, worldY, hash: base };
      }),
    [visiblePoints]
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

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setHoveredPoint(null);
  }

  useEffect(() => {
    if (view !== "point-cloud") return;
    resetView();
  }, [view, normalizedQuery, activeTopicId]);

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

  function handleMouseMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (!dragStart) return;
    const container = cloudRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const dx = (event.clientX - dragStart.x) / rect.width;
    const dy = (event.clientY - dragStart.y) / rect.height;
    setPan({ x: clamp(dragStart.panX + dx, -1.5, 1.5), y: clamp(dragStart.panY + dy, -1.5, 1.5) });
  }

  function showTooltip(event: ReactMouseEvent<HTMLAnchorElement>, point: PointCloudPoint) {
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
      <div className="px-6 pt-2">
        <p className="text-xs text-[var(--text-muted)]">
          {normalizedQuery ? `${view === "point-cloud" ? filteredPoints.length : lessonMatchCount} matching subtopics` : ""}
        </p>
      </div>

      {view === "point-cloud" ? (
        <div className="relative mx-3 mt-3 min-h-0 flex-1 overflow-hidden">
          <div
            ref={cloudRef}
            onWheel={handleWheel}
            onMouseDown={(event) => setDragStart({ x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y })}
            onMouseMove={handleMouseMove}
            onMouseUp={() => setDragStart(null)}
            onMouseLeave={() => {
              setDragStart(null);
              setHoveredPoint(null);
            }}
            className={`relative h-full w-full overflow-hidden ${
              dragStart ? "cursor-grabbing" : "cursor-grab"
            }`}
          >
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.08),transparent_35%),radial-gradient(circle_at_80%_70%,rgba(168,85,247,0.09),transparent_40%)]" />
            {pointCloudError ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                {pointCloudError}
              </div>
            ) : !pointCloudData || pointCloudData.points.length === 0 ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                Point cloud data is empty. Run `npm run build:point-cloud` first.
              </div>
            ) : visiblePoints.length === 0 ? (
              <div className="absolute inset-x-6 top-6 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)] p-3 text-sm text-[var(--text-muted)]">
                No matching subtopics found in point cloud.
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
                  {transformedPoints.length}/{visiblePoints.length} points - zoom {zoom.toFixed(2)}x
                </div>
                <button
                  type="button"
                  onClick={resetView}
                  className="absolute left-6 top-6 z-20 rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-2 py-1 text-[10px] text-[var(--text-muted)] transition hover:border-[var(--accent)] hover:text-[var(--text-strong)]"
                >
                  reset view
                </button>
                {transformedPoints.map((point) => (
                  <Link
                    key={`${point.topicId}:${point.lessonSlug}`}
                    href={topicHref(point.routeSlug, point.lessonSlug)}
                    title={`${point.topicTitle} - ${point.lessonTitle}`}
                    onMouseEnter={(event) => showTooltip(event, point)}
                    onMouseMove={(event) => showTooltip(event, point)}
                    onMouseLeave={() => setHoveredPoint(null)}
                    className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/20 shadow-[0_1px_8px_rgba(0,0,0,0.25)] transition hover:z-10 hover:scale-110 focus-visible:z-10 focus-visible:scale-110"
                    style={{
                      left: `${point.tx * 100}%`,
                      top: `${point.ty * 100}%`,
                      backgroundColor: topicColor(point.topicId),
                      width: "10px",
                      height: "10px",
                    }}
                    aria-label={`${point.topicTitle}: ${point.lessonTitle}`}
                  />
                ))}
                {hoveredPoint ? (
                  <div
                    className="pointer-events-none absolute z-30 max-w-[220px] rounded-md border border-[var(--border-strong)] bg-[var(--surface-1)] px-2 py-1.5 text-[11px] shadow-md"
                    style={{ left: hoveredPoint.x, top: hoveredPoint.y, transform: "translateY(-100%)" }}
                  >
                    <p className="truncate text-[var(--text-strong)]">{shortLabel(hoveredPoint.point.lessonTitle, 42)}</p>
                    <p className="truncate text-[10px] text-[var(--text-muted)]">{hoveredPoint.point.topicTitle}</p>
                  </div>
                ) : null}
              </>
            )}
          </div>

          <aside className="absolute left-6 top-16 z-20 max-h-[calc(100%-6rem)] w-[240px] overflow-auto rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)]/95 p-4 backdrop-blur-sm">
            <p className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">Topic key</p>
            <button
              type="button"
              onClick={() => setActiveTopicId(null)}
              className={`mt-3 w-full rounded-md border px-2 py-1 text-left text-xs transition ${
                activeTopicId === null
                  ? "border-[var(--accent)] bg-[var(--surface-2)] text-[var(--text-strong)]"
                  : "border-[var(--border-strong)] bg-[var(--surface-1)] text-[var(--text-muted)] hover:border-[var(--accent)]"
              }`}
            >
              Show all topics
            </button>
            <div className="mt-3 space-y-2">
              {pointTopicStats.length === 0 ? (
                <p className="text-sm text-[var(--text-muted)]">No topics available.</p>
              ) : (
                pointTopicStats.map((topic) => (
                  <button
                    key={topic.topicId}
                    type="button"
                    onClick={() => setActiveTopicId((current) => (current === topic.topicId ? null : topic.topicId))}
                    className={`flex w-full items-center justify-between gap-3 rounded-md border px-2 py-1 text-xs transition ${
                      activeTopicId === topic.topicId
                        ? "border-[var(--accent)] bg-[var(--surface-2)]"
                        : "border-[var(--border-strong)] hover:border-[var(--accent)]"
                    }`}
                  >
                    <span className="flex min-w-0 items-center gap-2">
                      <span
                        aria-hidden
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: topicColor(topic.topicId) }}
                      />
                      <span className="truncate text-[var(--text-strong)]">{topic.topicTitle}</span>
                    </span>
                    <span className="shrink-0 text-[var(--text-muted)]">
                      {topic.visible}/{topic.total}
                    </span>
                  </button>
                ))
              )}
            </div>
          </aside>
        </div>
      ) : filteredEntries.length === 0 ? (
        <div className="mx-6 mt-4 rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 text-center text-sm text-[var(--text-muted)]">
          No matching subtopics found.
        </div>
      ) : (
        <div className="mx-6 mt-4 min-h-0 flex-1 overflow-auto pr-1">
          <div className="grid gap-4 lg:grid-cols-2">
            {filteredEntries.map((entry) => (
              <div key={entry.routeSlug} className="rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-5">
                <Link
                  href={topicHref(entry.routeSlug)}
                  className="inline-flex rounded-md border border-[var(--border-strong)] bg-[var(--surface-2)] px-3 py-1 text-xs font-semibold uppercase tracking-wide text-[var(--accent)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]"
                >
                  {entry.meta.title}
                </Link>
                <p className="mt-3 text-sm text-[var(--text-muted)]">{entry.meta.description}</p>

                <div className="mt-4 flex flex-wrap gap-2">
                  {entry.lessons.map((lesson) => (
                    <Link
                      key={lesson.slug}
                      href={topicHref(entry.routeSlug, lesson.slug)}
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
