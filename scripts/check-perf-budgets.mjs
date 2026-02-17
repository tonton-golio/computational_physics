import fs from "node:fs";
import path from "node:path";

const projectRoot = process.cwd();
const buildManifestPath = path.join(projectRoot, ".next", "app-build-manifest.json");

const budgets = {
  "/topics": 900 * 1024,
  "/continuum-mechanics/page": 700 * 1024,
  "/complex-physics/page": 700 * 1024,
};

function formatKB(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

if (!fs.existsSync(buildManifestPath)) {
  console.error("Missing .next/app-build-manifest.json. Run `npm run build` first.");
  process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(buildManifestPath, "utf8"));
const pages = manifest.pages ?? {};
let hasFailure = false;

console.log("Route bundle budget check:");
for (const [route, budgetBytes] of Object.entries(budgets)) {
  const files = (pages[route] ?? []).filter((file) => file.endsWith(".js"));
  const totalBytes = files.reduce((sum, file) => {
    const filePath = path.join(projectRoot, ".next", file);
    if (!fs.existsSync(filePath)) return sum;
    return sum + fs.statSync(filePath).size;
  }, 0);

  const pass = totalBytes <= budgetBytes;
  const status = pass ? "PASS" : "FAIL";
  console.log(
    `- ${status} ${route}: ${formatKB(totalBytes)} (budget ${formatKB(budgetBytes)})`
  );

  if (!pass) hasFailure = true;
}

if (hasFailure) {
  console.error("Performance budget check failed.");
  process.exit(1);
}

console.log("All configured route budgets passed.");
