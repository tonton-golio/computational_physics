import fs from "fs";
import path from "path";

export interface TopicMeta {
  slug: string;
  title: string;
  description?: string;
  order?: number;
}

export interface TopicContent {
  slug: string;
  title: string;
  content: string;
  meta?: Record<string, string>;
}

// Get all slugs for a topic directory
export function getTopicSlugs(topicDir: string): string[] {
  const contentDir = path.join(process.cwd(), "content/topics", topicDir);
  
  if (!fs.existsSync(contentDir)) {
    return [];
  }
  
  const files = fs.readdirSync(contentDir);
  return files
    .filter((file) => file.endsWith(".md") || file.endsWith(".txt"))
    .map((file) => file.replace(/\.(md|txt)$/, ""));
}

// Get content for a specific topic file
export function getTopicFile(topicDir: string, slug: string): TopicContent | null {
  const basePath = path.join(process.cwd(), "content/topics", topicDir);
  
  // Try both .md and .txt extensions
  let filePath = path.join(basePath, `${slug}.md`);
  if (!fs.existsSync(filePath)) {
    filePath = path.join(basePath, `${slug}.txt`);
  }
  
  if (!fs.existsSync(filePath)) {
    return null;
  }
  
  const content = fs.readFileSync(filePath, "utf-8");
  
  // Extract title from first # heading
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1] : slug;
  
  // Parse frontmatter if present
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/);
  const meta: Record<string, string> = {};
  let bodyContent = content;
  
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    frontmatter.split("\n").forEach((line) => {
      const [key, ...valueParts] = line.split(":");
      if (key && valueParts.length > 0) {
        meta[key.trim()] = valueParts.join(":").trim();
      }
    });
    bodyContent = content.replace(frontmatterMatch[0], "");
  }
  
  return {
    slug,
    title: meta.title || title,
    content: bodyContent,
    meta,
  };
}

// Get all content files for a topic
export function getAllTopicFiles(topicDir: string): TopicContent[] {
  const slugs = getTopicSlugs(topicDir);
  return slugs
    .map((slug) => getTopicFile(topicDir, slug))
    .filter((item): item is TopicContent => item !== null);
}

// Topic configurations
export const TOPICS = {
  "quantum-optics": {
    title: "Quantum Optics",
    description: "Wigner functions, phase space dynamics, and quantum state visualization",
    difficulty: "Expert",
    color: "bg-purple-100 text-purple-800",
  },
  "continuum-mechanics": {
    title: "Continuum Mechanics",
    description: "Rod dynamics, iceberg simulations, and gravity waves",
    difficulty: "Expert",
    color: "bg-blue-100 text-blue-800",
  },
  "inverse-problems": {
    title: "Inverse Problems",
    description: "Tikhonov regularization and practical inversion techniques",
    difficulty: "Advanced",
    color: "bg-green-100 text-green-800",
  },
  "complex-physics": {
    title: "Complex Physics",
    description: "Statistical mechanics, percolation, and fractals",
    difficulty: "Intermediate",
    color: "bg-orange-100 text-orange-800",
  },
  "scientific-computing": {
    title: "Scientific Computing",
    description: "Linear equations, optimization, and numerical methods",
    difficulty: "Intermediate",
    color: "bg-cyan-100 text-cyan-800",
  },
  "online-reinforcement-learning": {
    title: "Online Reinforcement Learning",
    description: "Multi-armed bandits and regret analysis",
    difficulty: "Intermediate",
    color: "bg-pink-100 text-pink-800",
  },
  "advanced-deep-learning": {
    title: "Advanced Deep Learning",
    description: "GANs, VAEs, CNNs, and U-Net architectures",
    difficulty: "Advanced",
    color: "bg-red-100 text-red-800",
  },
  "applied-statistics": {
    title: "Applied Statistics",
    description: "Statistical methods and data analysis techniques",
    difficulty: "Beginner",
    color: "bg-yellow-100 text-yellow-800",
  },
  "dynamical-models": {
    title: "Dynamical Models",
    description: "Mathematical modeling of dynamic systems",
    difficulty: "Beginner",
    color: "bg-teal-100 text-teal-800",
  },
  "learn-to-code": {
    title: "Learn to Code",
    description: "Programming fundamentals",
    difficulty: "Beginner",
    color: "bg-gray-100 text-gray-800",
  },
} as const;

export type TopicSlug = keyof typeof TOPICS;
