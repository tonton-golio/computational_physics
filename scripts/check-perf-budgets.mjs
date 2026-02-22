import fs from "node:fs";
import path from "node:path";

const projectRoot = process.cwd();

// Budgets include all client JS for a route (framework + layout + page chunks).
// Next.js 16 Turbopack entryJSFiles reports the full set, unlike the old
// webpack app-build-manifest which only listed page-specific files.
const budgets = {
  "/topics": 2700 * 1024,
  "/topics/[topic]": 2700 * 1024,
  "/topics/[topic]/[slug]": 2700 * 1024,
};

// Map route paths to their manifest file locations and entryJSFiles keys.
// Next.js 16 replaced app-build-manifest.json with per-route
// page_client-reference-manifest.js files containing entryJSFiles.
const routeManifestMap = {
  "/topics": {
    manifestPath: ".next/server/app/topics/page_client-reference-manifest.js",
    entryKey: "[project]/src/app/topics/page",
  },
  "/topics/[topic]": {
    manifestPath:
      ".next/server/app/topics/[topic]/page_client-reference-manifest.js",
    entryKey: "[project]/src/app/topics/[topic]/page",
  },
  "/topics/[topic]/[slug]": {
    manifestPath:
      ".next/server/app/topics/[topic]/[slug]/page_client-reference-manifest.js",
    entryKey: "[project]/src/app/topics/[topic]/[slug]/page",
  },
};

function formatKB(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function getEntryJSFiles(manifestPath, entryKey) {
  const fullPath = path.join(projectRoot, manifestPath);
  if (!fs.existsSync(fullPath)) return null;

  const content = fs.readFileSync(fullPath, "utf8");
  globalThis.__RSC_MANIFEST = globalThis.__RSC_MANIFEST || {};
  eval(content);

  const manifest = Object.values(globalThis.__RSC_MANIFEST).pop();
  if (!manifest || !manifest.entryJSFiles) return null;

  return manifest.entryJSFiles[entryKey] ?? null;
}

// Verify at least one manifest exists (i.e. build has been run)
const anyManifestExists = Object.values(routeManifestMap).some(({ manifestPath }) =>
  fs.existsSync(path.join(projectRoot, manifestPath))
);

if (!anyManifestExists) {
  console.error(
    "Missing client reference manifests in .next/server/app/. Run `npm run build` first."
  );
  process.exit(1);
}

let hasFailure = false;

console.log("Route bundle budget check:");
for (const [route, budgetBytes] of Object.entries(budgets)) {
  const { manifestPath, entryKey } = routeManifestMap[route];
  const files = (getEntryJSFiles(manifestPath, entryKey) ?? []).filter((file) =>
    file.endsWith(".js")
  );

  const totalBytes = files.reduce((sum, file) => {
    const filePath = path.join(projectRoot, ".next", file);
    if (!fs.existsSync(filePath)) return sum;
    return sum + fs.statSync(filePath).size;
  }, 0);

  if (totalBytes === 0) {
    console.warn(`- WARN ${route}: no JS files found in manifest (key may be wrong)`);
    continue;
  }

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
