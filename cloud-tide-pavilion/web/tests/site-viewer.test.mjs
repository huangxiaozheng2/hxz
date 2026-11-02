// 展示网站的 3D 模型：载入、各预设视角、首层剖切、落潮都能切换且不报错；截图留在 tests/out 里对照
import fs from 'node:fs';
import path from 'node:path';
import { OUT, WEB, launch, open, reporter, serve } from './lib.mjs';

const R = reporter('展示网站 · 3D 模型');
fs.mkdirSync(OUT, { recursive: true });
const browser = await launch();
const srv = await serve(path.join(WEB, 'dist', 'site'));
const page = await open(browser, srv.url, { viewport: { width: 1100, height: 760 } });
await page.$eval('#viewer', e => e.scrollIntoView());
const t0 = Date.now();
await page.waitForFunction(() => !document.getElementById('status'), null, { timeout: 300000, polling: 500 });
R.ok('模型载入', true, `${((Date.now() - t0) / 1000).toFixed(1)} s（本机）`);
const views = await page.evaluate(() => [...document.querySelectorAll('button[data-view]')].map(b => b.id.replace('v-', '')));
for (const v of [...views, 'cut', 'low']) {
  const id = v === 'low' ? '#t-low' : '#v-' + v;
  await page.click(id);
  await page.waitForTimeout(v === 'low' ? 1500 : 3500);
  const bb = await (await page.$('#viewer')).boundingBox();
  await page.screenshot({ path: path.join(OUT, `site_${v}.png`), clip: bb, timeout: 240000 });
  if (v === 'cut') await page.click('#v-cut');
}
R.ok(`${views.length} 个预设视角、首层剖切、落潮都切换过`, views.length >= 5, views.join(' '));
R.ok('没有页面错误', !page.errors.length, page.errors.join(' | '));
await browser.close();
srv.close();
R.done();
