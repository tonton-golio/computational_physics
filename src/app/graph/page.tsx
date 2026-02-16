'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

// Starfield background component
function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);
    
    // Generate stars
    const stars: { x: number; y: number; size: number; opacity: number; speed: number }[] = [];
    const numStars = 200;
    
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 0.5,
        opacity: Math.random() * 0.8 + 0.2,
        speed: Math.random() * 0.02 + 0.01,
      });
    }
    
    let animationId: number;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      stars.forEach(star => {
        // Twinkle effect
        star.opacity += Math.sin(Date.now() * star.speed) * 0.01;
        star.opacity = Math.max(0.1, Math.min(1, star.opacity));
        
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity})`;
        ctx.fill();
      });
      
      animationId = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}

// Types
interface TopicInfo {
  id: string;
  title: string;
  description: string;
  difficulty: string;
  lessons: string[];
}

interface LessonContent {
  slug: string;
  title: string;
  content: string;
}

// Topic colors
const TOPIC_COLORS: Record<string, string> = {
  'quantum-optics': '#8b5cf6',
  'continuum-mechanics': '#3b82f6',
  'inverse-problems': '#22c55e',
  'complex-physics': '#f97316',
  'scientific-computing': '#06b6d4',
  'online-reinforcement-learning': '#ec4899',
  'advanced-deep-learning': '#ef4444',
  'applied-statistics': '#eab308',
  'dynamical-models': '#14b8a6',
  'learn-to-code': '#6b7280',
};

// Get node positions
function getNodePositions(
  topics: TopicInfo[],
  selectedTopic: string | null,
  viewMode: 'overview' | 'topic' | 'content'
): Record<string, { x: number; y: number; scale: number; opacity: number }> {
  const positions: Record<string, { x: number; y: number; scale: number; opacity: number }> = {};
  
  const centerX = typeof window !== 'undefined' ? window.innerWidth / 2 : 500;
  const centerY = typeof window !== 'undefined' ? window.innerHeight / 2 : 400;
  
  if (viewMode === 'overview') {
    topics.forEach((topic, index) => {
      const angle = (index / topics.length) * Math.PI * 2 - Math.PI / 2;
      const radius = Math.min(220, 150 + topics.length * 5);
      positions[topic.id] = {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        scale: 1,
        opacity: 1,
      };
    });
  } else if (viewMode === 'topic' && selectedTopic) {
    const topic = topics.find(t => t.id === selectedTopic);
    if (!topic) return positions;
    
    // Selected topic at center
    positions[topic.id] = { x: centerX, y: centerY - 80, scale: 1.15, opacity: 1 };
    
    // Lessons around it
    topic.lessons.forEach((lesson, index) => {
      const angle = (index / topic.lessons.length) * Math.PI * 2 - Math.PI / 2;
      const radius = 160;
      positions[`${topic.id}-${lesson}`] = {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius + 30,
        scale: 0.9,
        opacity: 1,
      };
    });
    
    // Other topics (dimmed)
    topics.forEach((t) => {
      if (t.id !== selectedTopic) {
        const index = topics.findIndex(x => x.id === t.id);
        const angle = (index / topics.length) * Math.PI * 2 - Math.PI / 2;
        const radius = 320;
        positions[t.id] = {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          scale: 0.6,
          opacity: 0.3,
        };
      }
    });
  }
  
  return positions;
}

// Tooltip
function Tooltip({ 
  title,
  description,
  color,
  lessonCount,
  position 
}: { 
  title: string;
  description: string;
  color: string;
  lessonCount?: number;
  position: { x: number; y: number };
}) {
  return (
    <div 
      className="absolute z-50 px-4 py-3 bg-[#1a1a2e] border border-[#2d2d44] rounded-xl shadow-2xl max-w-xs pointer-events-none"
      style={{ 
        left: position.x + 16, 
        top: position.y - 10,
        transform: 'translateY(-50%)'
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <div 
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <div className="text-white font-semibold">{title}</div>
        {lessonCount !== undefined && (
          <span className="text-xs text-[#4b5563]">{lessonCount} lessons</span>
        )}
      </div>
      <div className="text-[#9ca3af] text-sm leading-relaxed">{description}</div>
    </div>
  );
}

// Topic Node
function TopicNode({ 
  id,
  title,
  color,
  position,
  isFocused,
  onClick,
  onHover,
}: { 
  id: string;
  title: string;
  color: string;
  position: { x: number; y: number; scale: number; opacity: number };
  isFocused: boolean;
  onClick: () => void;
  onHover: (id: string | null) => void;
}) {
  const [isHovered, setIsHovered] = useState(false);
  
  const handleMouseEnter = () => {
    setIsHovered(true);
    onHover(id);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
    onHover(null);
  };
  
  return (
    <div
      className="absolute cursor-pointer transition-all duration-500 ease-out"
      style={{
        left: position.x,
        top: position.y,
        transform: `translate(-50%, -50%) scale(${position.scale})`,
        opacity: position.opacity,
      }}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div 
        className={`relative px-5 py-2.5 rounded-xl transition-all duration-300`}
        style={{ 
          backgroundColor: isHovered ? color + '25' : color + '15',
          border: `1.5px solid ${isHovered || isFocused ? color + '80' : color + '40'}`,
          boxShadow: isFocused ? `0 0 30px ${color}30` : isHovered ? `0 0 20px ${color}20` : 'none'
        }}
      >
        <div 
          className="absolute left-3 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full transition-all duration-300"
          style={{ 
            backgroundColor: color,
            opacity: isHovered || isFocused ? 1 : 0.6
          }}
        />
        <span 
          className="text-white font-medium whitespace-nowrap pl-2 transition-all duration-300"
          style={{ opacity: isHovered || isFocused ? 1 : 0.85 }}
        >
          {title}
        </span>
      </div>
    </div>
  );
}

// Lesson Node
function LessonNode({ 
  slug,
  position,
  onClick,
  onHover,
}: { 
  slug: string;
  position: { x: number; y: number; scale: number; opacity: number };
  onClick: () => void;
  onHover: (id: string | null) => void;
}) {
  const [isHovered, setIsHovered] = useState(false);
  
  // Format slug to title
  const title = slug
    .replace(/[-_]/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
  
  const handleMouseEnter = () => {
    setIsHovered(true);
    onHover(slug);
  };
  
  const handleMouseLeave = () => {
    setIsHovered(false);
    onHover(null);
  };
  
  return (
    <div
      className="absolute cursor-pointer transition-all duration-500 ease-out"
      style={{
        left: position.x,
        top: position.y,
        transform: `translate(-50%, -50%) scale(${position.scale})`,
        opacity: position.opacity,
      }}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div 
        className={`relative px-4 py-2 rounded-lg transition-all duration-300`}
        style={{ 
          backgroundColor: isHovered ? '#1e1e2e' : '#151525',
          border: `1.5px solid ${isHovered ? '#3b82f6' : '#2d2d44'}`,
        }}
      >
        <span className="text-[#9ca3af] text-sm">{title}</span>
      </div>
    </div>
  );
}

// Sidebar
function Sidebar({ 
  topic,
  lessons,
  selectedLesson,
  onSelectLesson,
  onBack,
}: { 
  topic: TopicInfo;
  lessons: LessonContent[];
  selectedLesson: LessonContent | null;
  onSelectLesson: (lesson: LessonContent) => void;
  onBack: () => void;
}) {
  const color = TOPIC_COLORS[topic.id] || '#3b82f6';
  
  return (
    <div className="absolute left-0 top-0 h-full w-72 bg-[#0f0f1a] border-r border-[#1e1e2e] flex flex-col">
      <div className="p-5 border-b border-[#1e1e2e]">
        <button 
          onClick={onBack}
          className="flex items-center gap-2 text-[#6b7280] hover:text-white mb-4 transition-colors text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span>Back to Overview</span>
        </button>
        
        <div className="flex items-center gap-2.5">
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
          <h2 className="text-lg font-semibold text-white">{topic.title}</h2>
        </div>
        <p className="text-[#6b7280] text-sm mt-2 leading-relaxed">{topic.description}</p>
        
        <div className="flex items-center gap-2 mt-3">
          <span className="text-xs text-[#4b5563]">{topic.lessons.length} lessons</span>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4">
        <div className="text-xs font-medium text-[#4b5563] uppercase tracking-wider mb-3 px-1">
          Lessons
        </div>
        <div className="space-y-1">
          {lessons.map((lesson) => (
            <button
              key={lesson.slug}
              onClick={() => onSelectLesson(lesson)}
              className={`w-full text-left px-3 py-2.5 rounded-lg transition-all ${
                selectedLesson?.slug === lesson.slug
                  ? 'bg-[#1e1e2e] border border-[#2d2d44]'
                  : 'hover:bg-[#151525] border border-transparent'
              }`}
            >
              <span className={`text-sm ${
                selectedLesson?.slug === lesson.slug ? 'text-white' : 'text-[#9ca3af]'
              }`}>
                {lesson.title}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// Content Panel with KaTeX rendering
function ContentPanel({ 
  lesson, 
  topicColor 
}: { 
  lesson: LessonContent;
  topicColor: string;
}) {
  // Render LaTeX with KaTeX
  const renderLatex = useCallback((latex: string, displayMode: boolean = false): string => {
    try {
      return katex.renderToString(latex, {
        displayMode,
        throwOnError: false,
        trust: true,
      });
    } catch {
      return `<span class="text-red-400">${latex}</span>`;
    }
  }, []);

  // Parse and render content
  const renderedContent = useMemo(() => {
    let content = lesson.content;
    
    // Handle graph embeds: {{graph:type}} or {{graph:harmonic-motion}}
    content = content.replace(/\{\{graph:([^}]+)\}\}/g, (_, graphType) => {
      const graphs: Record<string, { title: string; params: string }> = {
        'harmonic-motion': { title: 'Simple Harmonic Motion', params: 'A=1, ω=2' },
        'wave-propagation': { title: 'Wave Propagation', params: 'k=1, ω=1' },
        'gaussian': { title: 'Gaussian Distribution', params: 'μ=0, σ=1' },
        'field': { title: 'Vector Field', params: 'interactive' },
      };
      const graph = graphs[graphType] || { title: graphType, params: '' };
      return `<div class="my-6 p-4 bg-[#151525] rounded-lg border border-[#2d2d44]">
        <div class="flex justify-between items-center mb-3">
          <span class="text-sm text-white font-medium">${graph.title}</span>
          <span class="text-xs text-[#4b5563]">${graph.params}</span>
        </div>
        <div class="h-48 bg-[#0a0a14] rounded flex items-center justify-center text-[#4b5563] text-sm">
          [Interactive visualization - ${graphType}]
        </div>
      </div>`;
    });
    
    // First, handle display math ($$...$$)
    content = content.replace(/\$\$([^$]+)\$\$/g, (_, latex) => {
      return `<div class="my-4 p-4 bg-[#151525] rounded-lg overflow-x-auto">${renderLatex(latex, true)}</div>`;
    });
    
    // Then handle inline math ($...$)
    content = content.replace(/\$([^$\n]+)\$/g, (_, latex) => {
      return renderLatex(latex, false);
    });
    
    // Headers
    content = content.replace(/^##\s+__([^_]+)__/gm, '<h2 class="text-xl font-semibold text-white mt-8 mb-4">$1</h2>');
    content = content.replace(/^##\s+(.+)$/gm, '<h2 class="text-xl font-semibold text-white mt-8 mb-4">$1</h2>');
    content = content.replace(/^#\s+(.+)$/gm, '<h1 class="text-2xl font-bold text-white mt-6 mb-4">$1</h1>');
    
    // Bold and italic
    content = content.replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white font-medium">$1</strong>');
    content = content.replace(/\*([^*]+)\*/g, '<em class="text-[#d1d5db]">$1</em>');
    content = content.replace(/__([^_]+)__/g, '<strong class="text-white font-medium">$1</strong>');
    
    // Code blocks
    content = content.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="my-4 p-4 bg-[#151525] rounded-lg overflow-x-auto"><code class="text-sm text-[#e5e7eb]">$2</code></pre>');
    
    // Inline code
    content = content.replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-[#151525] rounded text-[#e5e7eb] text-sm">$1</code>');
    
    // Lists
    content = content.replace(/^- (.+)$/gm, '<li class="text-[#9ca3af] ml-4 mb-2">• $1</li>');
    
    // Paragraphs (split by double newline)
    content = content.split(/\n\n+/).map(block => {
      block = block.trim();
      if (block.startsWith('<h') || block.startsWith('<div') || block.startsWith('<pre') || block.startsWith('<li')) {
        return block;
      }
      return `<p class="text-[#9ca3af] leading-relaxed mb-4">${block.replace(/\n/g, '<br/>')}</p>`;
    }).join('\n');
    
    return content;
  }, [lesson.content, renderLatex]);
  
  return (
    <div className="flex-1 bg-[#0a0a14] overflow-y-auto">
      <div className="max-w-3xl mx-auto p-8 pt-12">
        <div className="flex items-center gap-3 mb-6">
          <div 
            className="w-4 h-4 rounded-full"
            style={{ backgroundColor: topicColor }}
          />
          <h1 className="text-3xl font-bold text-white">{lesson.title}</h1>
        </div>
        
        <div 
          className="prose prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: renderedContent }}
        />
      </div>
    </div>
  );
}

// Connection lines
function ConnectionLines({ 
  positions, 
  selectedTopic,
  lessons,
}: { 
  positions: Record<string, { x: number; y: number; scale: number; opacity: number }>;
  selectedTopic: string | null;
  lessons: string[];
}) {
  const lines: { from: [number, number]; to: [number, number] }[] = [];
  
  if (selectedTopic && positions[selectedTopic]) {
    const centerPos = positions[selectedTopic];
    lessons.forEach((lesson) => {
      const lessonPos = positions[`${selectedTopic}-${lesson}`];
      if (centerPos && lessonPos) {
        lines.push({
          from: [centerPos.x, centerPos.y],
          to: [lessonPos.x, lessonPos.y],
        });
      }
    });
  }
  
  return (
    <svg className="absolute inset-0 pointer-events-none" style={{ width: '100%', height: '100%' }}>
      {lines.map((line, i) => {
        const midY = (line.from[1] + line.to[1]) / 2;
        return (
          <path
            key={i}
            d={`M ${line.from[0]} ${line.from[1]} Q ${line.from[0]} ${midY}, ${line.to[0]} ${line.to[1]}`}
            fill="none"
            stroke="#2d2d44"
            strokeWidth="1.5"
            strokeDasharray="4 4"
            className="transition-opacity duration-500"
            opacity={0.6}
          />
        );
      })}
    </svg>
  );
}

export default function GraphPage() {
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [viewMode, setViewMode] = useState<'overview' | 'topic' | 'content'>('overview');
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedLesson, setSelectedLesson] = useState<LessonContent | null>(null);
  const [lessons, setLessons] = useState<LessonContent[]>([]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [positions, setPositions] = useState<Record<string, { x: number; y: number; scale: number; opacity: number }>>({});
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Fetch topics on mount
  useEffect(() => {
    fetch('/api/content')
      .then(res => res.json())
      .then(data => {
        setTopics(data.topics);
        setLoading(false);
      });
  }, []);
  
  // Filter topics based on search
  const filteredTopics = useMemo(() => {
    if (!searchQuery.trim()) return topics;
    const query = searchQuery.toLowerCase();
    return topics.filter(t => 
      t.title.toLowerCase().includes(query) || 
      t.description.toLowerCase().includes(query) ||
      t.lessons.some(l => l.toLowerCase().includes(query))
    );
  }, [topics, searchQuery]);
  
  // Fetch lessons when topic selected
  useEffect(() => {
    if (selectedTopic) {
      fetch(`/api/content?topic=${selectedTopic}`)
        .then(res => res.json())
        .then(data => {
          setLessons(data.lessons || []);
        });
    }
  }, [selectedTopic]);
  
  // Update positions
  useEffect(() => {
    const updatePositions = () => {
      const newPositions = getNodePositions(filteredTopics, selectedTopic, viewMode);
      setPositions(newPositions);
    };
    
    updatePositions();
    window.addEventListener('resize', updatePositions);
    return () => window.removeEventListener('resize', updatePositions);
  }, [filteredTopics, selectedTopic, viewMode]);
  
  const handleNodeClick = useCallback((nodeId: string, isLesson: boolean) => {
    if (viewMode === 'overview' && !isLesson) {
      setSelectedTopic(nodeId);
      setViewMode('topic');
    } else if (viewMode === 'topic' && isLesson) {
      const lesson = lessons.find(l => l.slug === nodeId);
      if (lesson) {
        setSelectedLesson(lesson);
        setViewMode('content');
      }
    } else if (viewMode === 'topic' && !isLesson && nodeId !== selectedTopic) {
      setSelectedTopic(nodeId);
      setSelectedLesson(null);
    }
  }, [viewMode, selectedTopic, lessons]);
  
  const handleBack = useCallback(() => {
    if (viewMode === 'content') {
      setViewMode('topic');
      setSelectedLesson(null);
    } else if (viewMode === 'topic') {
      setViewMode('overview');
      setSelectedTopic(null);
      setLessons([]);
    }
  }, [viewMode]);
  
  const currentTopic = topics.find(t => t.id === selectedTopic);
  const currentColor = currentTopic ? TOPIC_COLORS[currentTopic.id] : '#3b82f6';
  
  // Get hovered node info
  const hoveredTopic = hoveredNode ? filteredTopics.find(t => t.id === hoveredNode) : null;
  
  if (loading) {
    return (
      <div className="h-screen w-screen bg-[#0a0a14] flex items-center justify-center">
        <div className="text-[#4b5563] text-sm">Loading content...</div>
      </div>
    );
  }
  
  return (
    <div 
      className="h-screen w-screen bg-[#0a0a14] overflow-hidden relative font-sans"
      onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
    >
      {/* Starfield background */}
      <Starfield />
      
      {/* Search bar */}
      {viewMode !== 'content' && (
        <div className="absolute top-8 left-1/2 -translate-x-1/2 z-20">
          <div className="relative">
            <svg 
              className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#4b5563]" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search physics concepts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-80 pl-11 pr-4 py-2.5 bg-[#0f0f1a] border border-[#1e1e2e] rounded-xl text-white placeholder-[#4b5563] focus:outline-none focus:border-[#3b82f6] transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#4b5563] hover:text-white"
              >
                ✕
              </button>
            )}
          </div>
        </div>
      )}
      
      {/* Instructions */}
      {viewMode === 'overview' && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 text-[#4b5563] text-sm">
          Click a topic to explore • Hover for details
        </div>
      )}
      
      {/* Graph view */}
      {viewMode !== 'content' && (
        <div className="absolute inset-0">
          {/* Connection lines (only in topic view) */}
          {viewMode === 'topic' && selectedTopic && (
            <ConnectionLines 
              positions={positions} 
              selectedTopic={selectedTopic}
              lessons={lessons.map(l => l.slug)}
            />
          )}
          
          {/* Topic nodes */}
          {filteredTopics.map((topic) => {
            const pos = positions[topic.id];
            if (!pos) return null;
            
            return (
              <TopicNode
                key={topic.id}
                id={topic.id}
                title={topic.title}
                color={TOPIC_COLORS[topic.id] || '#3b82f6'}
                position={pos}
                isFocused={topic.id === selectedTopic}
                onClick={() => handleNodeClick(topic.id, false)}
                onHover={setHoveredNode}
              />
            );
          })}
          
          {/* Lesson nodes */}
          {viewMode === 'topic' && selectedTopic && lessons.map((lesson) => {
            const pos = positions[`${selectedTopic}-${lesson.slug}`];
            if (!pos) return null;
            
            return (
              <LessonNode
                key={lesson.slug}
                slug={lesson.slug}
                position={pos}
                onClick={() => handleNodeClick(lesson.slug, true)}
                onHover={setHoveredNode}
              />
            );
          })}
        </div>
      )}
      
      {/* Content view */}
      {viewMode === 'content' && currentTopic && (
        <div className="absolute inset-0 flex">
          <Sidebar
            topic={currentTopic}
            lessons={lessons}
            selectedLesson={selectedLesson}
            onSelectLesson={setSelectedLesson}
            onBack={handleBack}
          />
          {selectedLesson && <ContentPanel lesson={selectedLesson} topicColor={currentColor} />}
        </div>
      )}
      
      {/* Tooltip */}
      {hoveredTopic && viewMode === 'overview' && (
        <Tooltip 
          title={hoveredTopic.title}
          description={hoveredTopic.description}
          lessonCount={hoveredTopic.lessons.length}
          color={TOPIC_COLORS[hoveredTopic.id] || '#3b82f6'}
          position={mousePos}
        />
      )}
    </div>
  );
}
