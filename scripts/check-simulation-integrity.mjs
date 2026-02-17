import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const CONTENT_ROOT = path.join(ROOT, "content", "topics");
const VIS_ROOT = path.join(ROOT, "src", "components", "visualization");

function walkFiles(dir, extensions, acc = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(fullPath, extensions, acc);
      continue;
    }
    if (extensions.some((ext) => entry.name.endsWith(ext))) {
      acc.push(fullPath);
    }
  }
  return acc;
}

function extractPlaceholderIds() {
  const ids = new Set();
  const files = walkFiles(CONTENT_ROOT, [".md", ".txt"]);
  const pattern = /\[\[simulation\s+([^\]]+)\]\]/g;

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf-8");
    let match = pattern.exec(raw);
    while (match) {
      ids.add(match[1].trim());
      match = pattern.exec(raw);
    }
  }
  return ids;
}

function extractRegistryIds() {
  const ids = new Set();
  const files = walkFiles(VIS_ROOT, ["Simulations.tsx"]);
  const idPattern = /'([^']+)'\s*:/g;

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf-8");
    let match = idPattern.exec(raw);
    while (match) {
      ids.add(match[1]);
      match = idPattern.exec(raw);
    }
  }
  return ids;
}

const placeholders = extractPlaceholderIds();
const registered = extractRegistryIds();
const missing = [...placeholders].filter((id) => !registered.has(id)).sort();

if (missing.length > 0) {
  console.error("Simulation integrity check failed. Missing simulation ids:");
  for (const id of missing) {
    console.error(`- ${id}`);
  }
  process.exit(1);
}

console.info(`Simulation integrity check passed (${placeholders.size} placeholders, ${registered.size} registry ids).`);
