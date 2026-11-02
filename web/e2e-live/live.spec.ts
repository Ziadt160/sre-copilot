import { expect, test, type Page } from "@playwright/test";

// The whole path for real: Supabase login, /chat over SSE, the agent graph, an approval
// stored in Postgres, the resumed run calling mcp-ops with a minted token, the simulator.
const SHOTS = process.env.E2E_SCREENSHOTS;

test.beforeEach(async ({ page }) => {
  test.skip(!process.env.E2E_EMAIL || !process.env.E2E_PASSWORD, "set E2E_EMAIL and E2E_PASSWORD");
  await page.goto("/");
  await page.getByLabel("Email").fill(process.env.E2E_EMAIL ?? "");
  await page.getByLabel("Password").fill(process.env.E2E_PASSWORD ?? "");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByLabel("Message")).toBeVisible();
});

async function ask(page: Page, text: string) {
  await page.getByLabel("Message").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
}

async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: true });
}

test("a documentation question streams an answer with sources", async ({ page }) => {
  await ask(page, "What is a Pod?");
  await expect(page.getByTestId("answer")).not.toBeEmpty();
  await expect(page.getByRole("link").filter({ hasText: /./ }).first()).toHaveAttribute(
    "href",
    /kubernetes\.io/,
  );
  await shot(page, "answer-with-sources");
});

test("a restart waits for approval, then runs once on the cluster", async ({ page }) => {
  await ask(page, "The cart deployment in shop keeps getting OOM killed, restart it.");
  const dialog = page.getByRole("dialog", { name: /Approve action: restart_deployment/ });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("cart");
  await shot(page, "approval-dialog");
  await dialog.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByTestId("operation").last()).toContainText("succeeded");
  await expect(page.getByTestId("answer").last()).not.toBeEmpty();
  await shot(page, "operation-succeeded");
});

test("a rejected scale runs nothing", async ({ page }) => {
  await ask(page, "Scale the web deployment in shop to 4 replicas.");
  const dialog = page.getByRole("dialog", { name: /Approve action: scale_deployment/ });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Reject" }).click();
  await expect(page.getByTestId("answer").last()).toContainText(/reject/i);
  await expect(page.getByTestId("operation")).toHaveCount(0);
});
