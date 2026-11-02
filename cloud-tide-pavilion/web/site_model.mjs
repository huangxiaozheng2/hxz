// 展示网站模型预处理（之后再交给 gltf-transform optimize 压缩）：
//   1. 去掉用不上的贴图坐标：只有展廊的画用贴图，其余网格的 UV 白占体积；
//   2. 逐个网格按绝对误差简化（默认 1 cm）：meshopt 的误差是相对网格尺寸的，统一用一个比例时，
//      合在一起、跨度上百米的网格（如全场人物）会被简化得过狠，几米长的小构件却几乎没动；
//   3. 人物、金属柱与斜撑不简化：脸、手与发型体块经不起简化；全金属的圆管靠反射显出明暗，
//      截面少几条棱，反射就从天空变成湖面，整根发黑。
// 用法：node site_model.mjs 输入.glb 输出.glb [误差 m]
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { prune, simplifyPrimitive, weld } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';

const [src, dst, errArg] = process.argv.slice(2);
const ABS_ERR = Number(errArg || 0.01);
const KEEP = /^People_|_Columns$|YStrut|RingBeams/;

await MeshoptSimplifier.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(src);
await doc.transform(weld());

const count = p => (p.getIndices() ? p.getIndices().getCount() : p.getAttribute('POSITION').getCount()) / 3;
let before = 0, after = 0, uv = 0;
for (const mesh of doc.getRoot().listMeshes()) {
  const prims = mesh.listPrimitives();
  for (const p of prims) {
    const m = p.getMaterial();
    if (p.getAttribute('TEXCOORD_0') && !(m && m.getBaseColorTexture())) { p.setAttribute('TEXCOORD_0', null); uv++; }
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of prims) {
    const pos = p.getAttribute('POSITION'), mn = pos.getMin([]), mx = pos.getMax([]);
    for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], mn[i]); hi[i] = Math.max(hi[i], mx[i]); }
  }
  const size = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  for (const p of prims) {
    before += count(p);
    if (!KEEP.test(mesh.getName()) && p.getIndices()) {
      simplifyPrimitive(p, { simplifier: MeshoptSimplifier, ratio: 0, error: ABS_ERR / size, lockBorder: true });
    }
    after += count(p);
  }
}
await doc.transform(prune());
await io.write(dst, doc);
console.log(`site model: ${Math.round(before)} → ${Math.round(after)} triangles (error ${ABS_ERR * 100} cm), UV removed from ${uv} primitives`);
