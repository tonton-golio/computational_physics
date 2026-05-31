import { test, expect } from "@playwright/test";

test("topics route loads and shows navigation cards", async ({ page }) => {
  await page.goto("/topics");
  await expect(page.getByRole("link", { name: "Complex Physics" })).toBeVisible();
});
