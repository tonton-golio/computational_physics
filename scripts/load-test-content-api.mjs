const baseUrl = process.env.BASE_URL ?? "http://127.0.0.1:3000";
const totalRequests = Number(process.env.LOAD_TEST_REQUESTS ?? 200);
const concurrency = Number(process.env.LOAD_TEST_CONCURRENCY ?? 20);

const targets = [
  "/api/content",
  "/api/content?topic=continuum-mechanics",
  "/api/content?topic=complex-physics",
];

async function run() {
  const latencies = [];
  let ok = 0;
  let failed = 0;

  const workers = Array.from({ length: concurrency }, async (_, workerIdx) => {
    for (let i = workerIdx; i < totalRequests; i += concurrency) {
      const endpoint = targets[i % targets.length];
      const start = performance.now();
      try {
        const res = await fetch(`${baseUrl}${endpoint}`);
        const elapsed = performance.now() - start;
        latencies.push(elapsed);
        if (res.ok) {
          ok += 1;
        } else {
          failed += 1;
        }
      } catch {
        const elapsed = performance.now() - start;
        latencies.push(elapsed);
        failed += 1;
      }
    }
  });

  await Promise.all(workers);
  latencies.sort((a, b) => a - b);

  const p = (percentile) => {
    if (latencies.length === 0) return 0;
    const idx = Math.min(
      latencies.length - 1,
      Math.floor((percentile / 100) * latencies.length)
    );
    return latencies[idx];
  };

  console.log("Content API load test summary:");
  console.log(`- Base URL: ${baseUrl}`);
  console.log(`- Requests: ${totalRequests}`);
  console.log(`- Concurrency: ${concurrency}`);
  console.log(`- Success: ${ok}`);
  console.log(`- Failed: ${failed}`);
  console.log(`- p50: ${p(50).toFixed(1)} ms`);
  console.log(`- p95: ${p(95).toFixed(1)} ms`);
  console.log(`- p99: ${p(99).toFixed(1)} ms`);

  if (failed > 0) {
    process.exit(1);
  }
}

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
