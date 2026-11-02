// 两个页面在电脑与手机、浅色与深色下：模型载入、没有横向溢出、图片都取得到、没有页面错误
import fs from 'node:fs';
import path from 'node:path';
import { OUT, WEB, launch, open, reporter, serve } from './lib.mjs';

const R = reporter('版面');
fs.mkdirSync(OUT, { recursive: true });
const browser = await launch();
const sizes = [['电脑', 1280, 900], ['手机', 390, 844]];
for (const [site, ready] of [['ar', () => document.getElementById('mv').loaded], ['site', () => !document.getElementById('status')]]) {
  const srv = await serve(path.join(WEB, 'dist', site));
  for (const [dev, w, h] of sizes) for (const scheme of ['light', 'dark']) {
    const tag = `${site === 'ar' ? 'AR 网页' : '展示网站'} · ${dev} · ${scheme === 'light' ? '浅色' : '深色'}`;
    const page = await open(browser, srv.url, { viewport: { width: w, height: h }, colorScheme: scheme, deviceScaleFactor: 1 });
    if (site === 'site') await page.$eval('#viewer', e => e.scrollIntoView());
    await page.waitForFunction(ready, null, { timeout: 300000, polling: 500 });
    await page.waitForTimeout(1200);
    const info = await page.evaluate(async () => {
      const imgs = [...document.querySelectorAll('img')].map(i => i.getAttribute('src')).filter(Boolean);
      const broken = [];
      for (const src of new Set(imgs)) { const r = await fetch(src); if (!r.ok) broken.push(src); }
      return { sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, imgs: imgs.length, broken };
    });
    R.ok(tag, info.sw <= info.cw && !info.broken.length && !page.errors.length,
         `宽 ${info.sw}/${info.cw}，图片 ${info.imgs} 张${info.broken.length ? '，取不到：' + info.broken.join(' ') : ''}${page.errors.length ? '，错误：' + page.errors.join(' | ') : ''}`);
    await page.screenshot({ path: path.join(OUT, `layout_${site}_${dev === '电脑' ? 'desk' : 'mob'}_${scheme}.png`), fullPage: false });
    await page.context().close();
  }
  srv.close();
}
await browser.close();
R.done();
