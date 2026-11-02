// 导览影片逐帧渲染：在无界面浏览器里打开 web/dist/film，逐帧调用 film.frame(t) 并截图（JPEG）。
// 已经存在的帧会跳过，中断后可接着渲染。
//   node film/render.mjs 输出目录 [--size 1920x1080] [--fps 30] [--from 秒] [--to 秒] [--times 1,2.5,…]
import fs from 'node:fs';
import path from 'node:path';
import { WEB, launch, open, serve } from '../tests/lib.mjs';

const args = process.argv.slice(2);
const out = args[0];
const opt = k => { const i = args.indexOf('--' + k); return i > 0 ? args[i + 1] : null; };
const [W, H] = (opt('size') || '1920x1080').split('x').map(Number);
const fps = Number(opt('fps') || 30);
fs.mkdirSync(out, { recursive: true });

const browser = await launch();
const srv = await serve(path.join(WEB, 'dist', 'film'));
const page = await open(browser, srv.url, { viewport: { width: W, height: H }, deviceScaleFactor: 1 });
await page.waitForFunction(() => window.film && window.film.isReady, null, { timeout: 600000, polling: 500 });
if (page.errors.length) { console.log('页面错误：', page.errors.join(' | ')); process.exit(1); }
const D = await page.evaluate(() => window.film.duration);
const times = opt('times') ? opt('times').split(',').map(Number)
  : Array.from({ length: Math.round(D * fps) }, (_, i) => i / fps)
      .filter(t => t >= Number(opt('from') || 0) && t < Number(opt('to') || 1e9));
const t0 = Date.now();
let done = 0;
for (const t of times) {
  const file = path.join(out, opt('times') ? `t${t.toFixed(2)}.jpg` : `f${String(Math.round(t * fps)).padStart(5, '0')}.jpg`);
  if (fs.existsSync(file)) continue;
  await page.evaluate(t => window.film.frame(t), t);
  await page.screenshot({ path: file, type: 'jpeg', quality: 93 });
  done++;
  if (done % 60 === 0) {
    const per = (Date.now() - t0) / done / 1000;
    console.log(`${done} 帧，${per.toFixed(2)} s/帧，到 ${t.toFixed(1)} s`);
  }
}
if (page.errors.length) console.log('页面错误：', page.errors.join(' | '));
console.log(`完成：${done} 帧，用时 ${((Date.now() - t0) / 60000).toFixed(1)} 分钟`);
await browser.close();
srv.close();
