// AR 网页的点击：模型上、空白处、天空、标注文字、停用的按钮、“已经在这里”、玻璃、手机轻点
import { click, launch, openAR, reporter, settle } from './lib.mjs';

const R = reporter('AR 网页 · 点击');
const browser = await launch();
const cam = page => page.evaluate(() => { const c = window.__nav.camNow(); return { C: c.C, T: c.T, lv: window.__nav.state.level, zone: window.__nav.state.zone }; });
const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const page = await openAR(browser);
const box = await (await page.$('#mv')).boundingBox();

// 全景：找一个紧挨模型的空白点和一个远离模型的空白点
await page.evaluate('window.__nav.goOverview()'); await settle(page);
const pts = await page.evaluate(([b]) => {
  const out = { near: null, far: null }, mv = document.getElementById('mv');
  for (let y = b.y + 20; y < Math.min(b.y + b.height, innerHeight) - 20 && !(out.near && out.far); y += 6)
    for (let x = b.x + 20; x < b.x + b.width - 20; x += 6) {
      if (document.elementFromPoint(x, y) !== mv || window.__nav.pick(x, y)) continue;
      let dmin = 1e9;
      for (let a = 0; a < 16; a++) for (const r of [8, 16, 24, 32, 48, 70])
        if (window.__nav.pick(x + r * Math.cos(a / 16 * 6.283), y + r * Math.sin(a / 16 * 6.283))) dmin = Math.min(dmin, r);
      if (!out.near && dmin <= 16) out.near = [x, y];
      if (!out.far && dmin > 70) out.far = [x, y];
    }
  return out;
}, [box]);
await click(page, ...pts.near); await page.waitForTimeout(60);
let flew = await page.evaluate(() => window.__nav.flying); await settle(page);
let after = await cam(page);
R.ok('模型边上的空白处 → 落到最近的模型上', flew && after.lv === 2, `→ ${after.zone}`);

await page.evaluate('window.__nav.goOverview()'); await settle(page);
await click(page, ...pts.far); await page.waitForTimeout(320);
const fb = await page.evaluate(() => ({ ring: !!document.querySelector('#mv .tap-ring.miss'), hint: document.getElementById('hint').textContent, lv: window.__nav.state.level }));
R.ok('远离模型的空白处 → 虚线涟漪 + 提示', fb.ring && /空白处/.test(fb.hint) && fb.lv === 0, `「${fb.hint}」`);
await page.waitForTimeout(2600);
const back = await page.evaluate(() => document.getElementById('hint').textContent);
R.ok('提示过一会儿换回操作提示', !/空白处/.test(back), `「${back}」`);

// 人视点（门厅）点天空 → 原地转头
await page.evaluate('window.__nav.goSpot("lobby")'); await settle(page);
await page.waitForFunction(() => window.__nav.alpha.every(a => a > 0.999), null, { timeout: 20000 });
const sky = await page.evaluate(([b]) => {
  for (let y = b.y + 15; y < b.y + b.height / 2; y += 8) for (let x = b.x + 15; x < b.x + b.width - 15; x += 8)
    if (document.elementFromPoint(x, y) === document.getElementById('mv') && !window.__nav.pick(x, y)) return [x, y];
  return null;
}, [box]);
const before = await cam(page);
await click(page, ...sky); await page.waitForTimeout(60);
flew = await page.evaluate(() => window.__nav.flying); await settle(page);
after = await cam(page);
const l0 = before.T.map((v, i) => v - before.C[i]), l1 = after.T.map((v, i) => v - after.C[i]);
const turned = Math.acos(Math.min(1, l0.reduce((s, v, i) => s + v * l1[i], 0) / Math.hypot(...l0) / Math.hypot(...l1))) * 180 / Math.PI;
R.ok('人视点点天空 → 原地转头', flew && after.lv === 3 && dist(before.C, after.C) < 2e-4 && turned > 3, `转了 ${turned.toFixed(0)}°`);

// 标注文字可点（桌面）：每次回到首层后重新取位置（镜头朝向会变）
await page.evaluate('window.__nav.goLayer(1)'); await settle(page);
const keys = await page.evaluate(() => [...document.querySelectorAll('#mv .hotspot:not([hidden])')].map(b => b.slot.replace('hotspot-', '')));
let hitTags = 0, tried = 0, folded = 0;
for (const key of keys) {
  await page.evaluate('window.__nav.goLayer(1)'); await settle(page); await page.waitForTimeout(300);
  const t = await page.evaluate(k => {
    const b = document.querySelector(`#mv .hotspot[slot="hotspot-${k}"]`), tag = b.querySelector('.tag');
    if (b.classList.contains('tag-hide')) return { folded: true };
    const r = tag.getBoundingClientRect(), x = r.x + r.width * 0.6, y = r.y + r.height / 2;
    return { x, y, top: document.elementFromPoint(x, y) === tag };
  }, key);
  if (t.folded) { folded++; continue; }
  if (!t.top) continue;
  tried++;
  await click(page, t.x, t.y); await page.waitForTimeout(80); await settle(page);
  const z = await page.evaluate(() => window.__nav.state.zone);
  if (z === key) hitTags++; else console.log(`  标注 ${key} 的文字点了去了 ${z}`);
}
R.ok('标注文字：点了就去那个位置', hitTags === tried && tried >= keys.length - 2, `${hitTags}/${tried} 点中，${folded} 个收起，共 ${keys.length} 个`);

// 显示标注：画面里没有编号标注时停用
const dis = {};
for (const [n, js] of [['全景', 'window.__nav.goOverview()'], ['首层', 'window.__nav.goLayer(1)'], ['人视点', 'window.__nav.goSpot("gallery")']]) {
  await page.evaluate(js); await settle(page);
  dis[n] = await page.evaluate(() => document.getElementById('labels').disabled);
}
R.ok('「显示标注」在全景与人视点停用，进层后可用', dis['全景'] && !dis['首层'] && dis['人视点'], JSON.stringify(dis));

// 已经在那里：点当前层 / 当前位置 / 全景时点「回到全景」→ 提示
const here = [];
await page.evaluate('window.__nav.goLayer(1)'); await settle(page);
await page.click('#layers li:nth-child(4) .layer-btn'); await page.waitForTimeout(250);
here.push(await page.evaluate(() => document.getElementById('hint').textContent));
await page.evaluate('window.__nav.goSpot("artE")'); await settle(page);
await page.evaluate(() => document.querySelector('.sub button[data-spot="artE"]').click()); await page.waitForTimeout(250);
here.push(await page.evaluate(() => document.getElementById('hint').textContent));
await page.evaluate('window.__nav.goOverview()'); await settle(page);
await page.click('#home'); await page.waitForTimeout(250);
here.push(await page.evaluate(() => document.getElementById('hint').textContent));
R.ok('已经在那里时点它 → 提示', here.every(h => /已经/.test(h)), here.join(' / '));

// 玻璃不挡点击
await page.evaluate('window.__nav.goLayer(1)'); await settle(page);
const glass = await page.evaluate(([b]) => {
  const mv = document.getElementById('mv'); let n = 0, through = 0;
  for (let y = b.y + 20; y < Math.min(b.y + b.height, innerHeight) - 20; y += 26) for (let x = b.x + 20; x < b.x + b.width - 20; x += 26) {
    const m = mv.materialFromPoint(x, y);
    if (!m || !/Glass_Clear|Glass_Balustrade/.test(m.name)) continue;
    n++;
    const h = window.__nav.pick(x, y);
    if (h && !/Glass_Clear|Glass_Balustrade/.test(h.name)) through++;
  }
  return { n, through };
}, [box]);
R.ok('玻璃不挡点击：点到的是玻璃后面的东西', glass.n > 0 && glass.through === glass.n, `${glass.through}/${glass.n}`);
R.ok('桌面：没有页面错误', !page.errors.length, page.errors.join(' | '));
await page.context().close(); page.srv.close();

// 手机：手指挪了几个像素的轻点照样算点；拖动 30 px 不算；编号圆点边上也点得中
const ph = await openAR(browser, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
const pb = await (await ph.$('#mv')).boundingBox();
await ph.evaluate('window.__nav.goLayer(1)'); await settle(ph);
const target = await ph.evaluate(([b]) => {
  for (let y = b.y + b.height * 0.45; y < b.y + b.height; y += 10) for (let x = b.x + b.width * 0.3; x < b.x + b.width * 0.7; x += 10)
    if (document.elementFromPoint(x, y) === document.getElementById('mv') && window.__nav.pick(x, y)) return [x, y];
  return null;
}, [pb]);
const cdp = await ph.context().newCDPSession(ph);
const touch = async moves => {                          // 按真实手指的时间打时间戳（测试环境里一帧很慢，不能用处理时刻）
  const t0 = Date.now() / 1000;
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: target[0], y: target[1] }], timestamp: t0 });
  for (const [i, [dx, dy]] of moves.entries())
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: target[0] + dx, y: target[1] + dy }], timestamp: t0 + 0.05 * (i + 1) });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [], timestamp: t0 + 0.05 * (moves.length + 1) });
  await ph.waitForTimeout(80);
  const fl = await ph.evaluate(() => window.__nav.flying); await settle(ph);
  return fl && (await ph.evaluate(() => window.__nav.state.level)) === 2;
};
const taps = [];
for (const mv of [[7, 0], [0, 7], [6, 6], [-9, 3]]) {
  await ph.evaluate('window.__nav.goLayer(1)'); await settle(ph);
  taps.push(`${mv}:${(await touch([mv])) ? 'ok' : 'NO'}`);
}
R.ok('手机：手指挪了 7～10 px 的轻点照样起飞', taps.every(t => t.endsWith('ok')), taps.join(' '));
await ph.evaluate('window.__nav.goLayer(1)'); await settle(ph);
R.ok('手机：拖动 30 px 是转动模型，不起飞', !(await touch([[12, 0], [30, 0]])));
await ph.evaluate('window.__nav.goLayer(1)'); await settle(ph);
const dot = await ph.evaluate(() => {
  const b = [...document.querySelectorAll('#mv .hotspot:not([hidden])')].find(b => !b.classList.contains('off'));
  const r = b.getBoundingClientRect();
  return [r.x + r.width / 2, r.y + r.height / 2, r.width, b.slot.replace('hotspot-', '')];
});
await ph.touchscreen.tap(dot[0] + dot[2] / 2 + 6, dot[1]); await ph.waitForTimeout(80); await settle(ph);
R.ok('手机：点在编号圆点边上 6 px 也算点中', (await ph.evaluate(() => window.__nav.state.zone)) === dot[3]);
R.ok('手机：没有页面错误', !ph.errors.length, ph.errors.join(' | '));
await browser.close();
R.done();
