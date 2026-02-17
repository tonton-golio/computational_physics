declare module 'plotly.js-dist' {
  export * from 'plotly.js';
}

import 'plotly.js';

declare module 'plotly.js' {
  interface PlotData {
    nbinsx?: number;
    histnorm?: string;
  }
}
