// 起飞不抖：模拟“点下去时手指带了几像素拖动”（model-viewer 的目标已被拖开 6°），紧接着发起一次飞行，
// 用虚拟时钟逐帧推进，镜头水平角不能先往拖动方向弹一下再回来
import { launch, openAR, reporter } from './lib.mjs';

const R = reporter('AR 网页 · 起飞平顺');
const virtualClock = () => {
  const realNow = performance.now.bind(performance), base = realNow();
  const origRaf = window.requestAnimationFrame.bind(window), origCancel = window.cancelAnimationFrame.bind(window);
  let vt = 0, q = [];
  window.__virtual = false;
  window.requestAnimationFrame = cb => { if (!window.__virtual) return origRaf(cb); q.push(cb); return 1e6 + q.length; };
  window.cancelAnimationFrame = id => { if (id < 1e6) origCancel(id); };
  performance.now = () => window.__virtual ? base + vt : realNow();
  window.__startVirtual = () => { vt = realNow() - base; window.__virtual = true; };
  window.__step = ms => { vt += ms; const cbs = q; q = []; for (const cb of cbs) { try { cb(base + vt); } catch (e) { console.error(e); } } };
};
const browser = await launch();
const page = await openAR(browser, { viewport: { width: 640, height: 480 } }, virtualClock);
await page.waitForTimeout(800);
const step = () => page.evaluate(() => { window.__step(16.7); return document.getElementById('mv').getCameraOrbit().theta * 57.2958; });
await page.evaluate(() => window.__startVirtual());
for (let i = 0; i < 6; i++) await step();
await page.evaluate(() => { const mv = document.getElementById('mv'), o = mv.getCameraOrbit(); mv.cameraOrbit = `${o.theta + 0.1}rad ${o.phi}rad ${o.radius}m`; });
const th = [await step(), await step()];
await page.evaluate(() => {
  const mv = document.getElementById('mv'), rc = mv.getBoundingClientRect();
  const h = window.__nav.pick(rc.left + rc.width * 0.5, rc.top + rc.height * 0.55);
  window.__nav.goPoint({ x: h.p[0], y: h.p[1], z: h.p[2] }, 'gallery', false);
});
for (let i = 0; i < 14; i++) th.push(await step());
const d = th.slice(1).map((v, i) => v - th[i]);
const signs = d.filter(v => Math.abs(v) > 0.002).map(Math.sign);
const reversals = signs.slice(1).filter((s, i) => s !== signs[i]).length;
const maxStep = Math.max(...d.map(Math.abs));
R.ok('飞行开始时镜头不往回弹', reversals === 0, `方向反转 ${reversals} 次`);
R.ok('单帧转角不跳', maxStep < 2, `最大单帧 ${maxStep.toFixed(2)}°`);
R.ok('没有页面错误', !page.errors.length, page.errors.join(' | '));
await browser.close();
R.done();
