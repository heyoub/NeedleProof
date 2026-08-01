import { expect, test } from '@playwright/test';

test('run ownership is isolated across browser sessions', async ({ browser }) => {
  const aliceContext = await browser.newContext();
  const bobContext = await browser.newContext();
  const alice = await aliceContext.newPage();
  const bob = await bobContext.newPage();

  await alice.goto('/');
  await alice.locator('#rehearsal-button').click();
  await expect(alice.locator('#answer-status')).toContainText('Authoritative', {
    timeout: 15_000,
  });
  const runId = new URL(alice.url()).searchParams.get('run');
  expect(runId).toMatch(/^run_[0-9a-f]{32}$/);

  const aliceStatus = await alice.evaluate(async (id) =>
    fetch(`/api/runs/${id}`).then((response) => response.status), runId);
  expect(aliceStatus).toBe(200);

  await bob.goto('/');
  const bobRead = await bob.evaluate(async (id) =>
    fetch(`/api/runs/${id}`).then((response) => response.status), runId);
  const bobCancel = await bob.evaluate(async (id) =>
    fetch(`/api/runs/${id}/cancel`, { method: 'POST' }).then((response) => response.status), runId);
  expect(bobRead).toBe(404);
  expect(bobCancel).toBe(404);

  await aliceContext.close();
  await bobContext.close();
});

test('terminal run survives refresh and evidence dialog restores focus', async ({ page }) => {
  await page.goto('/');
  await page.locator('#rehearsal-button').click();
  await expect(page.locator('#answer-status')).toContainText('Authoritative', {
    timeout: 15_000,
  });
  await page.reload();
  await expect(page.locator('#answer-status')).toContainText('Authoritative', {
    timeout: 10_000,
  });

  const evidenceButton = page.locator('.evidence-button').first();
  await evidenceButton.click();
  await expect(page.locator('#evidence-dialog')).toBeVisible();
  await expect(page.locator('#close-evidence')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.locator('#evidence-dialog')).not.toBeVisible();
  await expect(evidenceButton).toBeFocused();
});
