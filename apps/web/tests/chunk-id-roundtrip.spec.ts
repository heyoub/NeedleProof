import { expect, test } from '@playwright/test';

test('opaque uint64 chunk ID survives the browser quote-verification round trip', async ({
  page,
}) => {
  let submittedChunkId: unknown;
  let submittedCorpusVersion: unknown;
  page.on('request', (request) => {
    if (request.url().endsWith('/api/verify/quote')) {
      submittedChunkId = request.postDataJSON()?.chunk_id;
      submittedCorpusVersion = request.postDataJSON()?.corpus_version;
    }
  });

  await page.goto('/');
  await page.locator('#rehearsal-button').click();
  await expect(page.locator('#answer-status')).toContainText('Authoritative', {
    timeout: 15_000,
  });

  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith('/api/verify/quote'),
  );
  await page.locator('#quote-challenge').click();
  const response = await responsePromise;

  expect(response.status()).toBe(200);
  expect(typeof submittedChunkId).toBe('string');
  expect(submittedChunkId).toMatch(/^chk_[0-9a-f]{16}$/);
  expect(submittedCorpusVersion).toMatch(/^v_[0-9a-f]{16}$/);
  const numericId = BigInt(`0x${String(submittedChunkId).slice(4)}`);
  expect(numericId).toBeGreaterThan(BigInt(Number.MAX_SAFE_INTEGER));
  await expect(page.locator('#quote-challenge-result')).toContainText(
    'Rejected as expected',
  );
});
