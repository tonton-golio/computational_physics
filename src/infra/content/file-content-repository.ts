import fs from "fs";
import path from "path";
import { z } from "zod";
import { AppError } from "@/shared/errors/app-error";
import type { ContentDocument } from "@/shared/types/content";

const metaSchema = z.record(z.string(), z.string()).default({});

const docSchema = z.object({
  topicId: z.string().min(1),
  slug: z.string().min(1),
  title: z.string().min(1),
  content: z.string(),
  meta: metaSchema,
});

function contentDir(topicId: string): string {
  return path.join(process.cwd(), "content/topics", topicId);
}

function parseFrontmatter(raw: string): { meta: Record<string, string>; body: string } {
  const frontmatterMatch = raw.match(/^---\n([\s\S]*?)\n---\n/);
  if (!frontmatterMatch) {
    return { meta: {}, body: raw };
  }

  const meta: Record<string, string> = {};
  for (const line of frontmatterMatch[1].split("\n")) {
    const [key, ...valueParts] = line.split(":");
    if (!key || valueParts.length === 0) continue;
    meta[key.trim()] = valueParts.join(":").trim();
  }

  return {
    meta,
    body: raw.replace(frontmatterMatch[0], ""),
  };
}

function titleFromMarkdown(raw: string, fallback: string): string {
  const heading = raw.match(/^#\s+(.+)$/m);
  return heading?.[1] ?? fallback;
}

export function listLessonSlugs(topicId: string): string[] {
  const dir = contentDir(topicId);
  if (!fs.existsSync(dir)) return [];

  try {
    return fs
      .readdirSync(dir)
      .filter((file) => file.endsWith(".md") || file.endsWith(".txt"))
      .map((file) => file.replace(/\.(md|txt)$/, ""));
  } catch (error) {
    throw new AppError("CONTENT_READ_ERROR", "Unable to read topic directory.", 500, {
      topicId,
      cause: String(error),
    });
  }
}

export function readLessonDocument(topicId: string, slug: string): ContentDocument | null {
  const dir = contentDir(topicId);
  const mdPath = path.join(dir, `${slug}.md`);
  const txtPath = path.join(dir, `${slug}.txt`);
  const target = fs.existsSync(mdPath) ? mdPath : txtPath;

  if (!fs.existsSync(target)) return null;

  let raw = "";
  try {
    raw = fs.readFileSync(target, "utf-8");
  } catch (error) {
    throw new AppError("CONTENT_READ_ERROR", "Unable to read content file.", 500, {
      topicId,
      slug,
      cause: String(error),
    });
  }

  const { meta, body } = parseFrontmatter(raw);
  const draft: ContentDocument = {
    topicId,
    slug,
    title: meta.title || titleFromMarkdown(body, slug),
    content: body,
    meta,
  };

  const parsed = docSchema.safeParse(draft);
  if (!parsed.success) {
    throw new AppError("CONTENT_PARSE_ERROR", "Invalid content document shape.", 500, {
      topicId,
      slug,
      issues: parsed.error.issues,
    });
  }
  return parsed.data;
}
