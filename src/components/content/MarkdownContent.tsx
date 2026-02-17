"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { CodeEditor } from "../ui/code-editor";
import { SimulationHost } from "@/features/simulation/SimulationHost";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

// Error placeholder
function SimulationError({ id, type }: { id: string; type: string }) {
  return (
    <div className="flex h-64 w-full flex-col items-center justify-center rounded-xl border border-[var(--danger-border)] bg-[var(--danger-surface)] text-[var(--text-soft)]">
      <span className="mb-2 text-[var(--accent-strong)]">Unknown placeholder type</span>
      <span className="text-sm">[[{type} {id}]]</span>
    </div>
  );
}

const FIGURE_DEFS: Record<string, { src: string; caption: string }> = {
  'mri-scan': {
    src: 'https://nci-media.cancer.gov/pdq/media/images/428431-571.jpg',
    caption: 'MRI example from medical imaging.',
  },
  'spect-scan': {
    src: 'https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20170104105121/ri/590/picture/Brain_SPECT_with_Acetazolamide_Slices_thumb.jpg',
    caption: 'SPECT scan example.',
  },
  'seismic-tomography': {
    src: 'http://www.earth.ox.ac.uk/~smachine/cgi/images/welcome_fig_tomo_depth.jpg',
    caption: 'Seismic tomography example.',
  },
  'claude-shannon': {
    src: 'https://d2r55xnwy6nx47.cloudfront.net/uploads/2020/12/Claude-Shannon_2880_Lede.jpg',
    caption: 'Claude Shannon, pioneer of information theory.',
  },
  'climate-grid': {
    src: 'https://caltech-prod.s3.amazonaws.com/main/images/TSchneider-GClimateModel-grid-LES-NEWS-WEB.width-450.jpg',
    caption: 'Model parameterization illustration.',
  },
  'gaussian-process': {
    src: 'https://gowrishankar.info/blog/gaussian-process-and-related-ideas-to-kick-start-bayesian-inference/gp.png',
    caption: 'Gaussian process visualization.',
  },
  'vertical-fault-diagram': {
    src: '/figures/vertical-fault-diagram.svg',
    caption: 'Simplified geometry used in vertical-fault inversion.',
  },
  'glacier-valley-diagram': {
    src: '/figures/glacier-valley-diagram.svg',
    caption: 'Glacier valley cross-section used in thickness inversion.',
  },
  'eigen-applications': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/35/Principal_component_analysis_of_Mars_data.png',
    caption: 'Eigenvalue applications in dimensionality reduction and modal analysis.',
  },
  'multiplicity-diagram': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/3f/Jordan_block.svg',
    caption: 'Algebraic vs geometric multiplicity and defective structure intuition.',
  },
  'convergence-comparison': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/8/8c/NewtonIteration_Ani.gif',
    caption: 'Qualitative convergence-rate comparison across iterative eigensolvers.',
  },
  'qr-convergence-plot': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/7e/QR_decomposition.svg',
    caption: 'QR-iteration intuition: repeated factorizations drive triangular convergence.',
  },
  'gershgorin-example': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/e/e2/Gerschgorin_disks.svg',
    caption: 'Gershgorin discs give fast eigenvalue location bounds in the complex plane.',
  },
  'algorithm-comparison': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/7a/Complexitysubsets.svg',
    caption: 'Algorithm trade-offs: convergence speed, robustness, and computational cost.',
  },
  'adl-mlops-loop': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/5/59/MLOps_lifecycle.png',
    caption: 'MLOps lifecycle diagram from data to deployment and monitoring.',
  },
  'adl-word2vec-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/01/Word2vec.png',
    caption: 'Skip-gram/Word2Vec architecture overview.',
  },
  'adl-lstm-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/9/93/LSTM_Cell.svg',
    caption: 'LSTM cell architecture and gating structure.',
  },
  'adl-gru-architecture': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/3/37/GRU.svg',
    caption: 'GRU architecture highlighting update and reset gates.',
  },
  'adl-cnn-feature-map': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png',
    caption: 'CNN feature hierarchy from local kernels to semantic representations.',
  },
  'continuum-stress-tensor': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/1/17/Stress_tensor_cartesian.svg',
    caption: 'Cartesian stress tensor components used in continuum mechanics.',
  },
  'continuum-hookes-law': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/4/47/Stress_strain_curve.svg',
    caption: 'Stress-strain relationship and Hooke-law linear regime.',
  },
  'continuum-density-fluctuations': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/7/73/Brownianmotion5cells.png',
    caption: 'Continuum approximation intuition from density fluctuations at finite scale.',
  },
  'complex-percolation-video': {
    src: 'https://archive.org/download/percolation-2d-simulation/percolation-2d-simulation.mp4',
    caption: 'Percolation evolution animation for cluster formation and critical threshold intuition.',
  },
  'complex-sandpile-image': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/0d/Sandpile_model.gif',
    caption: 'Self-organized criticality in sandpile avalanches.',
  },
  'orl-cartpole-learning-image': {
    src: 'https://upload.wikimedia.org/wikipedia/commons/0/06/Cart-pole.svg',
    caption: 'CartPole control setup used for reinforcement learning experiments.',
  },
  'mdp-agent-environment-loop': {
    src: 'https://www.researchgate.net/publication/350130760/figure/fig2/AS:1002586224209929@1616046591580/The-agent-environment-interaction-in-MDP.png',
    caption: 'Agent-environment interaction loop for Markov decision processes.',
  },
};

// Render simulation based on type and id
function renderPlaceholder(type: string, id: string): React.ReactNode {
  if (type === "simulation") {
    return <SimulationHost id={id} />;
  }

  if (type === "figure") {
    const knownFigure = FIGURE_DEFS[id];
    if (knownFigure) {
      return (
        <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
          <Image
            src={knownFigure.src}
            alt={knownFigure.caption}
            width={1200}
            height={700}
            className="max-h-[440px] w-full rounded border border-[var(--border-strong)] bg-[var(--surface-1)] object-contain"
          />
          <div className="mt-2 text-xs text-[var(--text-soft)]">{knownFigure.caption}</div>
        </div>
      );
    }

    const lower = id.toLowerCase();
    const hasExt = /\.[a-z0-9]+$/i.test(id);
    const src = hasExt ? `/figures/${id}` : `/figures/${id}.png`;
    const isVideo = lower.endsWith(".mp4") || lower.endsWith(".webm");

    if (isVideo) {
      return (
        <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
          <video controls className="w-full rounded" src={src}>
            Your browser does not support embedded video.
          </video>
          <div className="mt-2 text-xs text-[var(--text-soft)]">Figure: {id}</div>
        </div>
      );
    }

    return (
      <div className="w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-4">
        <Image
          src={src}
          alt={`Figure ${id}`}
          width={1200}
          height={700}
          className="w-full rounded border border-[var(--border-strong)]"
        />
        <div className="mt-2 text-xs text-[var(--text-soft)]">Figure: {id}</div>
      </div>
    );
  }

  if (type === "code-editor") {
    // Parse the id for parameters: initialCode|expectedOutput|solution
    const [initialCode, expectedOutput, solution] = id.split('|').map(s => s.trim());
    return (
      <CodeEditor
        initialCode={initialCode || ""}
        expectedOutput={expectedOutput}
        solution={solution}
        isExercise={!!expectedOutput}
      />
    );
  }

  return <SimulationError id={id} type={type} />;
}

// Simple markdown to HTML converter with LaTeX and placeholder support
export function MarkdownContent({ content, className = "" }: MarkdownContentProps) {
  const [processedContent, setProcessedContent] = useState<{
    html: string;
    placeholders: Array<{ type: string; id: string; index: number }>;
  }>({ html: "", placeholders: [] });
  const [katex, setKatex] = useState<typeof import("katex") | null>(null);
  const [katexLoadFailed, setKatexLoadFailed] = useState(false);

  useEffect(() => {
    import("katex")
      .then((k) => setKatex(k))
      .catch(() => setKatexLoadFailed(true));
  }, []);

  useEffect(() => {
    if (!katex && !katexLoadFailed) return;

    let processed = content;
    const placeholders: Array<{ type: string; id: string; index: number }> = [];

    // Extract and replace placeholders with temporary markers
    let placeholderIndex = 0;

    processed = processed.replace(/\[\[(simulation|figure|code-editor)\s+([^\]]+)\]\]/g, (match, type, id) => {
      const marker = `%%PLACEHOLDER_${placeholderIndex}%%`;
      placeholders.push({ type, id: id.trim(), index: placeholderIndex });
      placeholderIndex++;
      return marker;
    });

    // Convert block LaTeX ($$...$$ or \[...\])
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
      try {
        return `<div class="katex-block">${katex ? katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        }) : math.trim()}</div>`;
      } catch {
        return `$$${math}$$`;
      }
    });

    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
      try {
        return `<div class="katex-block">${katex ? katex.renderToString(math.trim(), {
          displayMode: true,
          throwOnError: false,
        }) : math.trim()}</div>`;
      } catch {
        return `\\[${math}\\]`;
      }
    });

    // Convert inline LaTeX ($...$ or \(...\))
    processed = processed.replace(/\$([^$\n]+?)\$/g, (_, math) => {
      try {
        return `<span class="katex-inline">${katex ? katex.renderToString(math.trim(), {
          displayMode: false,
          throwOnError: false,
        }) : math.trim()}</span>`;
      } catch {
        return `$${math}$`;
      }
    });

    processed = processed.replace(/\\\(([^)]+?)\\\)/g, (_, math) => {
      try {
        return `<span class="katex-inline">${katex ? katex.renderToString(math.trim(), {
          displayMode: false,
          throwOnError: false,
        }) : math.trim()}</span>`;
      } catch {
        return `\\(${math}\\)`;
      }
    });

    // Convert headers (order matters: longest prefix first)
    processed = processed.replace(/^###### (.+)$/gm, '<h6 class="text-sm font-semibold mt-4 mb-1 text-white">$1</h6>');
    processed = processed.replace(/^##### (.+)$/gm, '<h5 class="text-sm font-semibold mt-4 mb-1 text-white">$1</h5>');
    processed = processed.replace(/^#### (.+)$/gm, '<h4 class="text-base font-semibold mt-5 mb-2 text-white">$1</h4>');
    processed = processed.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-2 text-white">$1</h3>');
    processed = processed.replace(/^## (.+)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-3 text-white">$1</h2>');
    processed = processed.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-4 text-white">$1</h1>');

    // Convert bold and italic (support both ** and __ for bold, allow multiline)
    processed = processed.replace(/\*\*([\s\S]+?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/__([\s\S]+?)__/g, '<strong>$1</strong>');
    processed = processed.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Convert links
    processed = processed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>');

    // Convert code blocks
    processed = processed.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-[#0a0a15] text-gray-100 rounded-lg p-4 my-4 overflow-x-auto text-sm"><code>$2</code></pre>');

    // Convert inline code
    processed = processed.replace(/`([^`]+)`/g, '<code class="bg-[#0a0a15] px-1.5 py-0.5 rounded text-sm font-mono text-blue-300">$1</code>');

    // Convert lists
    processed = processed.replace(/^- (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');
    processed = processed.replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc my-4 space-y-1">$&</ul>');

    // Convert numbered lists
    processed = processed.replace(/^\d+\. (.+)$/gm, '<li class="ml-4 text-gray-300">$1</li>');
    
    // Convert tables
    processed = processed.replace(/^\|(.+)\|$/gm, (match, content) => {
      const cells = content.split('|').map((c: string) => c.trim());
      const isHeader = cells.some((c: string) => c.match(/^-+$/));
      if (isHeader) return '';
      const tag = 'td';
      return `<tr>${cells.map((c: string) => `<${tag} class="border border-gray-700 px-3 py-2 text-gray-300">${c}</${tag}>`).join('')}</tr>`;
    });
    
    // Wrap tables (allow whitespace/blank lines between rows from separator removal)
    processed = processed.replace(/(<tr>.*<\/tr>\s*)+/g, '<table class="w-full border-collapse my-4 bg-[#0a0a15] rounded">$&</table>');
    
    // Add table headers (first row after <table>)
    processed = processed.replace(/<table([^>]*)>(<tr>.*?<\/tr>)/g, (match, attrs, firstRow) => {
      const headerRow = firstRow.replace(/<td/g, '<th').replace(/<\/td>/g, '</th>');
      return `<table${attrs}>${headerRow}`;
    });

    // Convert horizontal rules
    processed = processed.replace(/^---$/gm, '<hr class="border-gray-700 my-8" />');

    // Convert paragraphs (double newlines)
    processed = processed.split('\n\n').map(para => {
      if (para.trim() && !para.startsWith('<')) {
        return `<p class="my-4 text-gray-300 leading-relaxed">${para}</p>`;
      }
      return para;
    }).join('\n');

    setProcessedContent({ html: processed, placeholders });
  }, [content, katex, katexLoadFailed]);

  if (!processedContent.html) {
    return <div className={className}>Loading...</div>;
  }

  // Split content by placeholder markers and render with components
  const parts = processedContent.html.split(/%%PLACEHOLDER_(\d+)%%/);
  
  return (
    <div className={`markdown-content prose max-w-none ${className}`}>
      {parts.map((part, index) => {
        if (index % 2 === 0) {
          // Regular HTML content
          return (
            <div
              key={index}
              dangerouslySetInnerHTML={{ __html: part }}
            />
          );
        } else {
          // Placeholder component
          const placeholderIndex = parseInt(part);
          const placeholder = processedContent.placeholders[placeholderIndex];
          if (placeholder) {
            return (
              <div key={index} className="my-8">
                {renderPlaceholder(placeholder.type, placeholder.id)}
              </div>
            );
          }
          return null;
        }
      })}
    </div>
  );
}
