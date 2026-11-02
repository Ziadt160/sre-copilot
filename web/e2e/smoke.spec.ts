import { expect, test, type Page } from "@playwright/test";

const MOCK = "http://127.0.0.1:8099";

interface MockState {
  feedback: Array<Record<string, unknown>>;
  events: Array<Record<string, unknown>>;
  decisions: Array<Record<string, unknown>>;
}

async function mockState(page: Page): Promise<MockState> {
  const res = await page.request.get(`${MOCK}/__state`);
  return (await res.json()) as MockState;
}

test.beforeEach(async ({ page }) => {
  await page.request.post(`${MOCK}/__reset`);
  await page.goto("/");
  await page.getByLabel("Email").fill("operator@example.test");
  await page.getByLabel("Password").fill("not-a-real-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByLabel("Message")).toBeVisible();
});

async function ask(page: Page, text: string) {
  await page.getByLabel("Message").fill(text);
  await page.getByRole("button", { name: "Send" }).click();
}

test("a chat streams and renders with sources, and feedback posts", async ({ page }) => {
  await ask(page, "What is a Pod?");
  await expect(page.getByTestId("answer")).toContainText("smallest deployable unit");
  const source = page.getByRole("link", { name: "Pods" });
  await expect(source).toHaveAttribute("href", /kubernetes\.io/);

  await page.getByRole("button", { name: "Source Pods was useful" }).click();
  await page.getByRole("button", { name: "Bad answer" }).click();
  await page.getByLabel("What was wrong?").fill("missing detail");
  await page.getByRole("button", { name: "Send feedback" }).click();
  await page.getByRole("button", { name: "Copy" }).click();

  await expect.poll(async () => (await mockState(page)).feedback.length).toBeGreaterThanOrEqual(2);
  const state = await mockState(page);
  expect(state.feedback.some((f) => (f.sources as Record<string, boolean>)["101"] === true)).toBe(true);
  expect(state.feedback.some((f) => f.rating === -1 && f.reason === "missing detail")).toBe(true);
  expect(state.events.some((e) => e.kind === "copy")).toBe(true);
});

test("a restart request shows the approval dialog; approve runs the operation", async ({ page }) => {
  await ask(page, "Please restart the checkout deployment");
  const dialog = page.getByRole("dialog", { name: /Approve action: restart_deployment/ });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("deployment/checkout");
  await expect(dialog).toContainText("Risk: medium");
  await expect(dialog.getByTestId("countdown")).toHaveText(/1[45]:\d\d/);

  // keyboard-usable: focus is inside the dialog, Escape closes it without deciding
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await page.getByRole("button", { name: "Review approval" }).click();
  await expect(dialog).toBeVisible();

  await dialog.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByTestId("operation")).toContainText("succeeded");
  await expect(page.getByTestId("answer")).toContainText("Restarted deployment/checkout");
  expect((await mockState(page)).decisions).toHaveLength(1);
});

test("reject shows the message and runs nothing", async ({ page }) => {
  await ask(page, "restart checkout");
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", { name: "Reject" }).click();
  await expect(page.getByTestId("answer")).toContainText("rejected");
  await expect(page.getByTestId("operation")).toHaveCount(0);
  expect((await mockState(page)).decisions[0]).toMatchObject({ decision: "reject" });
});

test("escalation and abstain states render", async ({ page }) => {
  await ask(page, "possible outage, page someone");
  await expect(page.getByText(/Escalated to a human/)).toBeVisible();
  await ask(page, "something unknown");
  await expect(page.getByText(/No answer: The documentation does not cover this/)).toBeVisible();
});
