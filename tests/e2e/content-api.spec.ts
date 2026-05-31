import { test, expect } from "@playwright/test";

test.describe("Content API", () => {
  test("GET /api/content returns topic index", async ({ request }) => {
    const response = await request.get("/api/content");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.topics).toBeTruthy();
    expect(body.topics.length).toBeGreaterThanOrEqual(10);
    expect(response.headers()["x-correlation-id"]).toBeTruthy();
  });

  test("GET /api/content?topic=complex-physics returns lessons", async ({ request }) => {
    const response = await request.get("/api/content?topic=complex-physics");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.topic.id).toBe("complex-physics");
    expect(body.lessons.length).toBeGreaterThan(0);
  });

  test("GET /api/content?topic=complex-physics&slug=percolation returns single lesson", async ({ request }) => {
    const response = await request.get("/api/content?topic=complex-physics&slug=percolation");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.content.topicId).toBe("complex-physics");
    expect(body.content.slug).toBe("percolation");
    expect(body.content.title).toBeTruthy();
  });

  test("returns 404 for unknown topic+slug", async ({ request }) => {
    const response = await request.get("/api/content?topic=complex-physics&slug=nonexistent-slug-xyz");
    expect(response.status()).toBe(404);
    const body = await response.json();
    expect(body.error.code).toBe("NOT_FOUND");
  });

  test("returns 404 for unknown topic", async ({ request }) => {
    const response = await request.get("/api/content?topic=nonexistent-topic-xyz");
    expect(response.status()).toBe(404);
  });

  test("GET /api/content with no params returns the topic index", async ({ request }) => {
    const response = await request.get("/api/content");
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body.topics).toBeTruthy();
  });

  test("rejects a traversal slug via the allowlist", async ({ request }) => {
    const response = await request.get("/api/content?topic=scientific-computing&slug=../../../README");
    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error.code).toBe("BAD_REQUEST");
  });
});

test.describe("Content Export API", () => {
  test("GET /api/content/export with no topic returns 400", async ({ request }) => {
    const response = await request.get("/api/content/export");
    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error.code).toBe("BAD_REQUEST");
  });

  test("GET /api/content/export with unknown topic returns 404", async ({ request }) => {
    const response = await request.get("/api/content/export?topic=nonexistent-topic-xyz");
    expect(response.status()).toBe(404);
    const body = await response.json();
    expect(body.error.code).toBe("NOT_FOUND");
  });

  test("GET /api/content/export with a known topic returns the topic and its lessons", async ({ request }) => {
    const response = await request.get("/api/content/export?topic=complex-physics");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.topic).toBe("complex-physics");
    expect(Array.isArray(body.lessons)).toBeTruthy();
    expect(body.lessons.length).toBeGreaterThan(0);
  });
});
