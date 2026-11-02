// 展廊的画：从展廊、首层总览、画墙前点画面，都要走到那幅画所在的画墙前
import { click, launch, openAR, reporter, settle } from './lib.mjs';

const R = reporter('AR 网页 · 点画');
const WALL = ['artE', 'artE', 'artS', 'artS', 'artW', 'artW'];       // 画 0–5 挂在哪段画墙
const browser = await launch();
const page = await openAR(browser);
for (const [from, js] of [['展廊人视点', 'window.__nav.goSpot("gallery")'], ['首层总览', 'window.__nav.goLayer(1)'], ['西段画墙前', 'window.__nav.goSpot("artW")']]) {
  await page.evaluate(js); await settle(page);
  const r = await (await page.$('#mv')).boundingBox();
  const found = await page.evaluate(([r]) => {                     // 扫屏幕找一幅看得见、没被别的东西挡住的画
    for (let y = r.y + 20; y < Math.min(r.y + r.height, innerHeight) - 20; y += 9)
      for (let x = r.x + 20; x < r.x + r.width - 20; x += 9) {
        if (document.elementFromPoint(x, y) !== document.getElementById('mv')) continue;
        const h = window.__nav.pick(x, y);
        if (h && /^M_Art_\d/.test(h.name)) return [x, y, h.name];
      }
    return null;
  }, [r]);
  if (!found) { R.ok(`${from}：画面里找到画`, false); continue; }
  await click(page, found[0], found[1]); await page.waitForTimeout(100); await settle(page);
  const st = await page.evaluate(() => ({ zone: window.__nav.state.zone, level: window.__nav.state.level }));
  const want = WALL[Number(found[2].split('_')[2])];
  R.ok(`${from}：点 ${found[2]} → ${want}`, st.zone === want && st.level === 3, `到了 ${st.zone}`);
}
R.ok('没有页面错误', !page.errors.length, page.errors.join(' | '));
await browser.close();
R.done();
