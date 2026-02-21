export interface CanvasTheme {
  bg: string;
  grid: string;
  axis: string;
  text: string;
  textStrong: string;
  legendBg: string;
  font: string;
  fontTitle: string;
  fontSmall: string;
  fontAnnotation: string;
}

const FONT_STACK = 'ui-monospace, SFMono-Regular, Menlo, monospace';

export const DARK_THEME: CanvasTheme = {
  bg: '#101a2d',
  grid: '#1c2e4a',
  axis: '#2c4166',
  text: '#8fa3c4',
  textStrong: '#d0dbed',
  legendBg: 'rgba(16,26,45,0.85)',
  font: `11px ${FONT_STACK}`,
  fontTitle: `13px ${FONT_STACK}`,
  fontSmall: `10px ${FONT_STACK}`,
  fontAnnotation: `9px ${FONT_STACK}`,
};

export const LIGHT_THEME: CanvasTheme = {
  bg: '#edf0f5',
  grid: '#d4dae6',
  axis: '#a8b4ca',
  text: '#4a5a72',
  textStrong: '#1a2235',
  legendBg: 'rgba(237,240,245,0.9)',
  font: `11px ${FONT_STACK}`,
  fontTitle: `13px ${FONT_STACK}`,
  fontSmall: `10px ${FONT_STACK}`,
  fontAnnotation: `9px ${FONT_STACK}`,
};

export function getCanvasTheme(): CanvasTheme {
  if (typeof document === 'undefined') return DARK_THEME;
  return document.documentElement.getAttribute('data-theme') === 'light' ? LIGHT_THEME : DARK_THEME;
}
