# Figures Guide

## Purpose

Explain how to add static figures (images and videos) to lesson content, how the figure system resolves assets, and how the lightbox display works.

## Placeholder Syntax

In lesson markdown (`content/topics/<topic-id>/<slug>.md`), embed a figure with:

```md
[[figure figure-id]]
```

This placeholder is extracted by `src/lib/markdown-to-html.ts` and rendered by `src/components/content/MarkdownContent.tsx`.

## Resolution Order

When `MarkdownContent` encounters a `[[figure figure-id]]` placeholder, it resolves the asset in this order:

1. **Registry lookup** — checks `FIGURE_DEFS[figure-id]` for a registered figure with explicit `src` and `caption`.
2. **Convention-based fallback** — if not registered, derives the path:
   - If the id has a file extension (e.g. `diagram.svg`): `/figures/{id}`
   - If no extension: `/figures/{id}.png`
   - Video extensions (`.mp4`, `.webm`) render a `<video>` element instead of `<img>`

Prefer registering figures in the registry for proper captions and optimized `<Image>` rendering.

## Adding a Figure

### Step 1: Place the asset

Put the image or video file in `public/figures/`:

```
public/figures/my-diagram.svg
public/figures/my-photo.png
public/figures/my-animation.mp4
```

Supported formats: SVG, PNG, JPG, WebP (images); MP4, WebM (videos).

### Step 2: Register in the topic registry (recommended)

Each topic has a figures export in its `*Simulations.tsx` registry file:

```tsx
// src/components/visualization/InverseProblemsSimulations.tsx

export const INVERSE_FIGURES: Record<string, { src: string; caption: string }> = {
  'vertical-fault-diagram': {
    src: '/figures/vertical-fault-diagram.svg',
    caption: 'Simplified geometry used in vertical-fault inversion.',
  },
  'glacier-valley-diagram': {
    src: '/figures/glacier-valley-diagram.svg',
    caption: 'Glacier valley cross-section used in thickness inversion.',
  },
};
```

If the topic doesn't already export a figures object, add one and wire it into the aggregator at `src/lib/figure-definitions.ts`:

```tsx
// figure-definitions.ts
import { MY_TOPIC_FIGURES } from '@/components/visualization/MyTopicSimulations';

export const FIGURE_DEFS: Record<string, { src: string; caption: string }> = {
  ...MY_TOPIC_FIGURES,
  // ...other topic figures
};
```

### Step 3: Reference in markdown

```md
The diagram below shows the fault geometry:

[[figure vertical-fault-diagram]]

As we can see, the layers are horizontal...
```

### Quick path (no registration)

For rapid prototyping, skip the registry. Just place the file in `public/figures/` and use the placeholder:

```md
[[figure my-quick-diagram.svg]]
```

This skips the optimized `<Image>` component and renders a basic `<img>` tag without a proper caption. Register the figure before merging to main.

## Display Behavior

### FigureBlock wrapper

Every figure is rendered inside a `FigureBlock` component that provides:

- A rounded card with border and background (`bg-[var(--surface-2)]`)
- The caption as muted text below the image
- A hover-reveal expand button (top-right corner)
- Click-to-zoom behavior (images only, not videos)

### Lightbox

Clicking an image (or the expand button) opens `FigureLightbox`:

- Full-viewport dark overlay with the image centered
- Caption displayed as a frosted glass pill (top-left)
- Close button (top-right) or click outside to dismiss
- Inline suggestion box at the bottom for feedback
- Videos play inline with native controls (no lightbox expand)

## File Structure

| File | Role |
|---|---|
| `public/figures/` | Static figure assets (images, videos) |
| `src/lib/figure-definitions.ts` | Aggregator — merges all topic figure exports |
| `src/components/visualization/*Simulations.tsx` | Per-topic figure registries (`*_FIGURES` exports) |
| `src/components/content/MarkdownContent.tsx` | Renders `[[figure]]` placeholders as `FigureBlock` |
| `src/components/ui/figure-lightbox.tsx` | Fullscreen lightbox overlay |

## Currently Registered Figures

| Figure ID | Topic | Format |
|---|---|---|
| `vertical-fault-diagram` | inverse-problems | SVG |
| `glacier-valley-diagram` | inverse-problems | SVG |
| `aml-loss-outlier-comparison` | applied-machine-learning | SVG |
| `aml-tree-growth-steps` | applied-machine-learning | SVG |
| `aml-activation-gallery` | applied-machine-learning | SVG |
| `aml-numeric-forward-backward` | applied-machine-learning | SVG |

## Do / Don't

### Do

- register figures in the topic `*_FIGURES` export for proper captions and optimized rendering
- use SVG for diagrams and illustrations (scalable, small file size)
- use descriptive, kebab-case figure IDs that include the topic prefix (e.g. `aml-tree-growth-steps`)
- keep figure assets in `public/figures/` (flat directory, no subdirectories)
- write informative captions that explain what the figure shows

### Don't

- use the convention-based fallback for production content (always register)
- use PNG for diagrams that could be SVG
- duplicate figure IDs across topics (the aggregator merges all into one flat map)
- use spaces or uppercase in figure IDs
- put large video files (>5MB) in `public/figures/` without compression
