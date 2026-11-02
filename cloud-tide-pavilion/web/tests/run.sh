#!/bin/bash
# 依次运行全部浏览器测试（先 web/build.py 构建）。每个测试打印 PASS/FAIL，有失败时退出码为 1。
# 软件渲染下较慢，全部跑完约 15–20 分钟；也可以单独运行：node tests/ar-clicks.test.mjs
cd "$(dirname "$0")/.." || exit 1
status=0
for t in tests/*.test.mjs; do
  node "$t" || status=1
done
[ $status -eq 0 ] && echo "全部通过" || echo "有测试未通过"
exit $status
