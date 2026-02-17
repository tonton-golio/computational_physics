import { test, expect } from "@playwright/test";

test("topics route loads and shows navigation cards", async ({ page }) => {
  await page.goto("/topics");
  await expect(page.getByRole("heading", { name: "Explore Topics" })).toBeVisible();
});
