// 浏览器测试公用：本地静态服务（web/dist/…）、浏览器、把网页引用的 CDN 换成 node_modules 里同版本的文件
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const OUT = path.join(WEB, 'tests', 'out');                 // 截图等输出（不进仓库）
const NM = path.join(WEB, 'node_modules');

let pw;
try { pw = await import('playwright'); }
catch { pw = await import('/opt/node22/lib/node_modules/playwright/index.mjs'); }   // 云端开发环境里预装的位置

const TYPES = { '.html': 'text/html; charset=utf-8', '.json': 'application/json', '.js': 'text/javascript',
                '.webp': 'image/webp', '.jpg': 'image/jpeg', '.png': 'image/png' };
// 页面是片段：发布平台会补上文档骨架，这里同样补上
const wrap = body => '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" ' +
  'content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + body + '</body></html>';

export function serve(dir) {
  const root = path.resolve(dir);
  if (!fs.existsSync(path.join(root, 'index.html'))) throw new Error(`${root} 还没构建：先运行 web/build.py`);
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p.endsWith('/')) p += 'index.html';
    const f = path.join(root, p);
    if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); res.end(); return; }
    let body = fs.readFileSync(f);
    if (f.endsWith('.html')) body = Buffer.from(wrap(body.toString()));
    res.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream', 'content-length': body.length });
    res.end(body);
  });
  server.unref();                                     // 不因这个服务挂住进程：浏览器关掉后测试就能结束
  return new Promise(ok => server.listen(0, '127.0.0.1', () =>
    ok({ url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() })));
}

const CDN = [
  [/^https:\/\/cdn\.jsdelivr\.net\/npm\/@google\/model-viewer@4\.3\.1\/(.+)$/, '@google/model-viewer'],
  [/^https:\/\/cdn\.jsdelivr\.net\/npm\/meshoptimizer@0\.22\.0\/(.+)$/, 'meshoptimizer'],
  [/^https:\/\/cdn\.jsdelivr\.net\/npm\/three@0\.160\.0\/(.+)$/, 'three'],
];

export async function launch() {
  return pw.chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
}

// 打开一个页面：收集页面错误；字体请求直接拒绝（离线也能测）；CDN 脚本换成本地同版本文件
// init：页面脚本运行前注入的函数（如虚拟时钟）
export async function open(browser, url, ctxOpts = {}, init = null) {
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 800 }, ...ctxOpts });
  const page = await ctx.newPage();
  if (init) await page.addInitScript(init);
  page.errors = [];
  page.on('pageerror', e => page.errors.push(e.message));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  await page.route(/^https:\/\/cdn\.jsdelivr\.net\/npm\//, r => {
    const u = r.request().url();
    for (const [re, pkg] of CDN) {
      const m = re.exec(u);
      const f = m && path.join(NM, pkg, m[1]);
      if (f && fs.existsSync(f)) return r.fulfill({ path: f, contentType: 'text/javascript' });
    }
    console.log('  没有本地副本，已拦下：', u);
    return r.abort();
  });
  await page.goto(url);
  return page;
}

// AR 网页：等模型与点击网格都就绪，把模型滚到视窗中间
export async function openAR(browser, ctxOpts, init) {
  const srv = await serve(path.join(WEB, 'dist', 'ar'));
  const page = await open(browser, srv.url, ctxOpts, init);
  page.srv = srv;
  await page.waitForFunction(() => document.getElementById('mv').loaded && window.__nav && window.__nav.picker.ready,
                             null, { timeout: 180000 });
  await page.evaluate(() => document.getElementById('mv').scrollIntoView({ block: 'center' }));
  return page;
}

// 镜头停稳：不在飞行中，再等一会儿让各层淡入淡出走完
export async function settle(page, ms = 700) {
  await page.waitForFunction(() => !window.__nav.flying, null, { timeout: 60000 });
  await page.waitForTimeout(ms);
}

export async function click(page, x, y) {
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.waitForTimeout(60);
  await page.mouse.up();
}

// 结果记录：每项一行 PASS/FAIL，最后汇总；有失败时进程退出码为 1
export function reporter(title) {
  const rows = [];
  console.log(`# ${title}`);
  return {
    ok(name, cond, info = '') { rows.push(!!cond); console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${info ? '  ' + info : ''}`); },
    done() {
      const pass = rows.filter(Boolean).length;
      console.log(`${pass}/${rows.length} passed\n`);
      if (pass < rows.length) process.exitCode = 1;
    },
  };
}
