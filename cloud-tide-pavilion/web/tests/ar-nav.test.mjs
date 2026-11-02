// AR 网页的层次导航：用页面上的“层次”索引逐层、逐个位置点过去，检查状态、路径与说明卡；分层展开；Esc 返回
import { launch, openAR, reporter, settle } from './lib.mjs';

const R = reporter('AR 网页 · 层次导航');
const browser = await launch();
const page = await openAR(browser, { viewport: { width: 1280, height: 900 } });
const st = () => page.evaluate(() => ({
  level: window.__nav.state.level, layer: window.__nav.state.layer, zone: window.__nav.state.zone, explode: window.__nav.state.explode,
  crumbs: [...document.querySelectorAll('#crumbs li')].map(l => l.textContent).join(' › '),
  place: document.getElementById('placeName').textContent,
}));

let s = await st();
R.ok('打开时是全景', s.level === 0 && s.crumbs === '全景' && s.place === '云上潮汐馆', s.crumbs);

const layers = await page.evaluate(() => [...document.querySelectorAll('#layers > li')].map(li => li.querySelector('.lv-name').firstChild.textContent));
const bad = [];
let spots = 0;
for (let i = 0; i < layers.length; i++) {
  const k = layers.length - 1 - i;                          // 索引自上而下：第 1 行是最上层
  await page.click(`#layers > li:nth-child(${i + 1}) .layer-btn`); await settle(page, 900);
  s = await st();
  if (!(s.level === 1 && s.layer === k && s.place === layers[i])) bad.push(`层 ${k + 1}：${JSON.stringify(s)}`);
  const subs = await page.evaluate(n => [...document.querySelectorAll(`#layers > li:nth-child(${n}) .sub button`)]
    .map(b => [b.dataset.spot, b.querySelector('.t').textContent, !!b.querySelector('.eye')]), i + 1);
  for (const [key, name, eye] of subs) {
    await page.click(`#layers > li:nth-child(${i + 1}) .sub button[data-spot="${key}"]`); await settle(page, 900);
    s = await st();
    spots++;
    const ok = s.zone === key && s.level === (eye ? 3 : 2) && s.place === name && (!eye || s.crumbs.endsWith('人视点'));
    if (!ok) bad.push(`${name}：${JSON.stringify(s)}`);
  }
}
R.ok(`五层逐层进入、${spots} 个位置逐个飞到`, !bad.length && spots >= 15, bad.join(' | '));

await page.click('#home'); await settle(page, 900);
await page.click('#explode'); await settle(page, 1800);
s = await st();
const ex = await page.evaluate(() => [document.getElementById('explode').textContent, Math.max(...window.__nav.lift)]);
R.ok('分层展开：各层抬起，按钮变成「合拢各层」', s.explode && ex[0] === '合拢各层' && ex[1] > 0.1, `最高抬起 ${ex[1].toFixed(3)}`);
await page.click('#explode'); await settle(page, 1800);
const lifted = await page.evaluate(() => Math.max(...window.__nav.lift));
R.ok('再按一次合拢', !(await st()).explode && lifted < 1e-4);

await page.click('#layers > li:nth-child(4) .layer-btn'); await settle(page, 900);
await page.click('#layers > li:nth-child(4) .sub button[data-spot="gallery"]'); await settle(page, 900);
const path = [(await st()).level];
for (let i = 0; i < 3; i++) { await page.keyboard.press('Escape'); await settle(page, 900); path.push((await st()).level); }
R.ok('Esc 逐级返回：人视点 → 位置 → 层 → 全景', path.join('') === '3210', path.join(' → '));
R.ok('没有页面错误', !page.errors.length, page.errors.join(' | '));
await browser.close();
R.done();
