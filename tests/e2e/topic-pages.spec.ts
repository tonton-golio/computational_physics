import { test, expect } from "@playwright/test";

test.describe("Topic pages", () => {
  test("topic detail page loads", async ({ page }) => {
    await page.goto("/topics/complex-physics");
    await expect(page.locator("h1, h2").first()).toBeVisible();
  });

  test("lesson page loads content", async ({ page }) => {
    await page.goto("/topics/complex-physics/percolation");
    // Lesson pages should render markdown content
    await expect(page.locator("main, article, [role='main']").first()).toBeVisible();
  });

  test("homepage loads", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("body")).toBeVisible();
  });

  test("404 page for invalid topic", async ({ page }) => {
    const response = await page.goto("/topics/nonexistent-topic-xyz");
    expect(response?.status()).toBe(404);
  });
});
