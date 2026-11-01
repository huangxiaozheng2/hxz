# 云上潮汐馆 · Cloud Tide Pavilion

一座浮在水面上的小型展馆：三片月牙屋顶围绕一座会随潮涨落的庭院。模型用 Blender（bpy）以 Python 程序化生成，用 Cycles 渲染；分析图由同一套几何数据绘制。

![展板一](renders/board_01.jpg)
![展板二](renders/board_02.jpg)
![展板三](renders/board_03.jpg)
![展板四](renders/board_04.jpg)
![展板五](renders/board_05.jpg)
![展板六](renders/board_06.jpg)

## 设计说明

**区位与到达。** 设定场地为北纬 30° 的一处湖泊。展馆离西南岸约 67 m，由湖岸公园的入口广场经约 80 m 的栈桥到达。湖岸公园有石砌驳岸、6 m 宽的滨水步道、入口广场与通往滨湖路的园路；步道内侧一排垂柳，其后是成片的阔叶大树、灌木与松。岸线刻意避开冬至日落方向，夕阳仍落在开阔水面上。

**潮汐庭院。** 三片月牙屋顶交错环抱一座阶梯式潮池。潮池经西南的水口与外部水面相通，一日两涨两落：设计高潮位 ±0.00 时池水漫至第三级台阶，低潮位 -0.95 时整池露底、留下潮痕。树岛立在池中，磨砂玻璃雕塑立在池底石台上，高潮时下半段没入水中。

**从潮到云。** 参观路线自下而上：
1. 湖岸公园入口广场 → 80 m 栈桥 → 低矮门廊 → 门厅 → 视线直达潮池；
2. 一层展廊贴近水面，外墙为象牙白实墙，只开四处深框景窗；天光从屋面光缝顺墙洗下；
3. 1:12 无障碍坡道绕潮池上升约 1.96 圈，到达主屋顶上的屋面平台；
4. 次屋顶下的观景厅，西端隔着月洞墙看水和天。

**功能与疏散。** 展廊东端尖角为卫生间（含无障碍）与储藏，朝庭院一侧改为实墙；设备间与雨水蓄水池在其下的基座箱体内。屋面平台与观景厅有两个出口：1:12 坡道，以及展廊最宽处、电梯以西的封闭疏散楼梯间（钢筋混凝土筒，双跑，踏步 170 × 270，梯段净宽 1.2 m，首层直达屋面平台，两出口相距约 10 m）。观景厅设三处门：南门对坡道到达点，北门通东平台，西门通月洞平台与楼梯间。`program.py` 按 0.25 m 网格计算疏散距离（墙与玻璃不可穿越）：首层任一点到最近对外门步行 ≤ 16 m、直线 ≤ 15 m；屋面平台到最近出口步行 ≤ 23 m、直线 ≤ 12 m，均在 GB 50016 第 5.5.17 条“展览厅内任一点至最近疏散门直线距离 ≤ 30 m”以内。

**冬至日落轴线。** 场地按北纬 30° 计算：冬至日落方位 242.6°。水口、潮池与月洞墙都对准这个方向。冬至傍晚，夕阳沿水口照进潮池，同一时刻也穿过月洞落在观景厅里。

## 关键数据

| 项目 | 数值 |
|---|---|
| 室内面积 | 约 397 m²：展廊 233、门厅 60、卫生间 11、储藏 3、疏散楼梯间 20、电梯 4、观景厅 46；另有屋面平台 101 m²、基座内设备间 27 m² |
| 区位 | 设定场地：北纬 30° 湖泊；离西南岸约 67 m，栈桥约 80 m |
| 潮位 | 设计高潮位 ±0.00，低潮位 -0.95，潮差 0.95 m |
| 标高 | 首层 +0.85（高出高潮位 0.85 m），屋面平台 +6.30，主屋顶西翼尖 +11.2，次屋顶最高点约 +12.2 |
| 潮池 | 四级台阶（+0.55 / +0.25 / -0.05 / -0.35），池底 -0.65；水口净宽 2.6 m |
| 坡道 | 1:12，8 段，每段升 0.68 m，段间 1.5 m 休息平台，全长约 78 m；上下两圈净高约 2.5 m |
| 天光 | 屋面光缝 8 段，宽 0.7 m，沿展廊外墙内侧布置 |
| 日照 | 冬至日落方位 242.6°，渲染时太阳高度角 4° |

## 结构体系

**荷载路径。** 屋面板（轻钢龙骨 + 20 mm GRC）→ 100 mm 木格栅次肋 → 胶合木主梁（约 1.5 m 一根，与幕墙分格对齐）→ 柱顶钢箱环梁 → 柱 / 承重墙 / 电梯筒 / 疏散楼梯间 / Y 形斜柱 / 水中石墩 → 钻孔灌注桩。

**支承布置。**
- 主屋顶 A：展廊内侧柱列 + 西端高厅外侧柱列（每 3 个幕墙分格一根，约 4.5 m）、展廊北侧承重实墙、电梯筒、疏散楼梯间、东翼尖下一根补充柱；西翼尖由水中石墩上的 Y 形斜柱两肢托起。
- 次屋顶 B：二层柱立在主屋顶结构层上，与一层内侧柱上下对齐；另由电梯筒与疏散楼梯间（均贯通至 B）、月洞墙和观景厅东端弧墙支承。月洞墙与楼梯间旁由墙直接承托，不再设柱，所以月洞框景里看不到柱子。
- 低檐 C：门厅内外两侧柱列、西端一根补充柱；东翼尖下的基座已切除（净高不足处改为水面），由两处水中石墩上的钢柱承托。
- 屋面上任一点到最近竖向支承不超过 5.8 m（`structure.py` 计算，展板四左下图）。

**抗侧力与风。** 水平力由电梯筒、疏散楼梯间两处钢筋混凝土筒体与展廊北侧弧形承重墙、月洞墙承担；屋面板下设钢拉杆水平支撑，形成刚性隔板，把风力传给筒体；其余柱为铰接，只承竖向荷载。低檐 C 下没有墙，门厅柱柱脚刚接、柱顶与环梁刚接，形成钢框架。开阔水面上的大挑檐由风吸控制连接：柱头顶板与环梁之间设抗拔螺栓，主梁下缘由木格栅侧向约束，承受反向弯矩。

| 构件 | 做法 |
|---|---|
| 主梁 | 胶合木 GL28h，宽 160；高 200（檐口）→ 600 / 520 / 480（A / B / C 支座），截面随悬挑弯矩加深 |
| 次肋 | 木格栅 100 深，间距约 240 mm，端部统一收于阴影缝内线 |
| 柱顶环梁 | 钢箱梁 220 宽，梁底与主梁底平 |
| 柱 | 钢管 φ194×10，外张柱头，顶板 φ500；门厅柱柱脚刚接 |
| Y 形斜柱 | 变截面钢管 φ460 → φ320，铸钢节点，立于 φ2.1 m 石墩 |
| 电梯筒 | 钢筋混凝土筒 φ2.3 m，由基础贯通至次屋顶 |
| 疏散楼梯间 | 钢筋混凝土筒 2.9 × 7.0 m，墙厚 250，由基础直达次屋顶，墙顶承托次屋顶主梁 |
| 月洞墙 / 展廊外墙 | 钢筋混凝土承重墙，外饰矿物质涂层 |
| 水中石墩 | 花岗岩外包 φ1.24 m，高出设计高潮位 0.6 m |
| 坡道 | 螺旋钢箱梁 260 高，外侧 φ180 钢柱每 30° 一根，以钢悬臂托住经过的每一圈 |

**初步校核**（`structure.prelim_checks()`；屋面恒载 1.5 / 活载 0.5 kPa，屋面平台恒载 3.0 / 活载 3.5 kPa，分项系数 1.3 / 1.5）：

| 部位 | 跨度 / 悬挑 | 截面 | 应力比 | 挠度 |
|---|---|---|---|---|
| A 主梁悬挑 | 5.7 m | 160×600 | 0.35 | L/536 |
| B 主梁悬挑 | 5.3 m | 160×520 | 0.42 | L/412 |
| C 主梁悬挑 | 5.8 m | 160×480 | 0.58 | L/258 |
| 屋面平台主梁 | 7.3 m 简支 | 160×600 | 0.50 | L/739 |
| 一层柱（叠加二层柱） | 5.0 m | φ194×10 | 0.23（N = 305 kN，稳定承载 1355 kN） | — |

**风与抗侧力估算**（`structure.lateral_checks()`；基本风压 0.55 kPa，A 类地貌，风振系数统一取 1.2；地震 7 度 0.10 g）：

| 项目 | 结果 |
|---|---|
| 风吸 | 挑檐净风吸 1.22 kPa；按 1.0 恒载 + 1.5 风组合净上吸 0.33 kPa，主梁反弯应力比 0.07，柱头拉力约 7 kN |
| 门厅刚接柱（低檐 C） | 12 根柱分担约 119 kN 风力，柱底弯矩 44 kN·m，应力比 0.37；按悬臂柱计侧移 H/176（限值 H/150），柱顶与环梁刚接后会小得多 |
| 筒体抗剪（A / B） | 设计风剪力约 465 kN，大于多遇地震约 248 kN，由风控制；较弱方向（楼梯间两道短墙 + 电梯筒半截面）抗剪承载约为其 5.3 倍 |

悬挑一律按“到最近支承的距离”当作悬臂计算，偏保守。这些是方案阶段的手算，只用来确认构件尺度与体系是否合理；深化阶段需要结构工程师建模复核（风洞或规范风荷载取值、地震、木梁与钢环梁节点、筒体与桩基）。

## 构造

| 部位 | 做法 |
|---|---|
| 屋檐 | 楔形截面，边缘收薄到 0.15 m，半圆形拉丝金属包边；檐下 120 mm 阴影缝 |
| 排水 | 距檐口 0.45–0.75 m 一圈余弦形浅槽，内衬深色金属格栅；槽内设虹吸雨水斗，水平管走屋面结构层，立管藏于钢柱（内径 174 mm，内置 DN100 管）与承重墙内，汇入基座内的蓄水池，回用于冲厕与潮池补水；翼尖低点设溢流口，直接排向水面 |
| 展廊外墙 | 象牙白矿物质实墙，四处两跨宽的框景窗，外挑 0.62 m 的古铜色深窗框 |
| 幕墙 | 等宽分段约 1.5 m，顶部金属收口带封至木梁底，室内侧暗藏洗木梁的灯带 |
| 潮池石材 | 高潮位以下颜色加深、表面湿润，近池底泛淡青褐色，低潮时显出潮痕 |
| 栏杆 | 屋面平台、栈桥与首层平台临水边、湖岸步道（栈桥两侧各约 150 m）：12+12 夹胶玻璃分段（板缝 12 mm）+ φ48 圆管扶手 + 底部压条，高 1.05 m；在水口、斜柱石墩与上岸处断开 |

## 文件

```
cloud-tide-pavilion/
├── model/
│   ├── cloud_tide_pavilion.blend   完整场景（集合分组、材质、灯光、相机）
│   └── cloud_tide_pavilion.glb     建筑、庭院、室内陈设、人物与湖岸配景（简化 PBR 材质；配景树叶抽稀，人物减面、发丝以发型体块代替，不含水面与远山）
├── renders/                        各机位渲染图；分析图（区位、总平面、首层与平台平面、1-1 与 2-2 剖面、南立面、1:20 檐口节点、形体生成、环境策略、潮位、支承与抗侧力）；结构轴测图；立面线稿底图 elevation_south_raw.png；六张展板 board_01–06.jpg
└── src/
    ├── plan.py          全部几何定义：三片屋顶（含主梁高度）、围合体、潮池与水口、坡道、日落轴线、天窗、疏散楼梯间、服务用房、湖岸与栈桥
    ├── facade.py        幕墙分格、实墙段、框景窗与门的位置
    ├── structure.py     结构体系：柱网、承重墙、二层柱、支承距离检查、初步校核、风与抗侧力估算
    ├── program.py       功能与疏散：房间面积、出口、任一点到最近出口的步行与直线距离
    ├── architecture.py  屋顶与天窗、胶合木主次梁、幕墙与框景窗、基座与潮池、平台、坡道、电梯筒、疏散楼梯间、卫生间隔墙、柱 / 环梁 / 斜柱 / 石墩、螺旋楼梯、月洞墙
    ├── scenery.py       树岛与雕塑、室内陈设与画作射灯、家具、人物布置、湖岸公园（驳岸、步道、广场、垂柳与阔叶树丛、灌木、地灯、坐凳；树按变体共享网格）、岛屿松树、远山
    ├── people.py        人物：头、躯干、四肢与手（含手指）的体块并集经体素重建，关节无接缝，脸有鼻、眉弓、唇与耳；衣物为外扩外壳加垂坠褶皱；头发为发丝曲线（毛发 BSDF），按短发、齐耳、披肩、马尾、发髻生长并受重力下垂
    ├── materials.py     材质库（统一以 M_ 开头）
    ├── scene.py         天空（物理天空 + 程序化薄云）、灯光、相机（每个机位带曝光和潮位）
    ├── build.py         总装 → .blend
    ├── render.py        渲染指定机位；按相机切换涨潮/落潮水位，dusk 机位切换为黄昏
    ├── diagrams.py      区位图、总平面与日落轴线、首层与屋面平台平面（含疏散路线）、1-1 长剖面与 2-2 横剖面、南立面（在线稿底图上加水体、标高与标注）、1:20 檐口节点、形体生成、环境策略、潮位曲线、支承距离与抗侧力图
    ├── struct_axo.py    结构轴测图（屋面半透明，承重构件着色）与标注锚点
    ├── elevation.py     立面线稿底图：正交相机 + Freestyle 线稿 + 浅色材质，透明底；栈桥在台基南缘外剖切
    ├── boards.py        六张 A1 展板排版（概念、技术、平面与流线、结构、生成与环境、立面与构造）
    └── export_glb.py    导出 GLB（网页版由 gltf-transform 以 meshopt 压缩，保留构件名，便于网页中按构件剖切）
```

### 场景分组（Outliner）

```
CloudTidePavilion
├── Architecture
│   ├── Roofs            Roof_A_Main / Roof_B_Upper / Roof_C_Low（_Shell、_EdgeTrim、_SoffitRibs、_PrimaryBeams_Glulam，A 另有 _Skylight_*）
│   ├── Envelopes        Envelope_E1_Gallery / E2_Lobby / E3_Belvedere（_Glass、_Mullions_Frames、_SolidWall、_WindowFrames、_CoveLED、_Columns）、Service_WC_Store
│   ├── Plinth_Terraces  Plinth_And_Bridge、Plinth_EdgeRailings（临水栏杆）、Courtyard_TidalBasin、Roof_Terrace_Deck、Terrace_MoonGate
│   ├── Structure        Structure_LiftCore、Structure_EgressStair（EgressStair_Walls / Flights / FireDoors）、Structure_Supports（Supports_Columns / RingBeams_Steel / YStrut_Steel / StonePiers）
│   └── Circulation      Courtyard_Ramp、Hall_SpiralStair
├── Site                 Courtyard、Interior_Exhibits、Lights_GalleryAccent、Terrace_Furniture、People_ScaleFigures（People_*_i 网格、People_HairStrands_i 发丝曲线、People_HairProxy_Web 仅供网页）、Landscape（Water_Surface、Shore_*、Islets、Mountains_*）
├── Lights
└── Cameras              Cam_hero / approach / arrival / axis / axis_out / moon / court_ht / court_lt / aerial / interior / gallery / terrace / stair / structure / detail / dusk
```

## 重新生成

```bash
python -m venv venv && ./venv/bin/pip install bpy==5.2.2 shapely triangle scipy pillow matplotlib
cd cloud-tide-pavilion
../venv/bin/python src/build.py model/cloud_tide_pavilion.blend
../venv/bin/python src/render.py model/cloud_tide_pavilion.blend renders hero,approach,arrival,axis,moon,court_ht,court_lt,aerial,axis_out,interior,gallery,terrace,stair,structure,detail,dusk --samples=128
../venv/bin/python src/struct_axo.py model/cloud_tide_pavilion.blend renders/struct_axo.png --samples=64
../venv/bin/python src/elevation.py model/cloud_tide_pavilion.blend renders/elevation_south_raw.png --side=south --samples=48   # Freestyle 线稿较慢，约 20 分钟
../venv/bin/python src/diagrams.py renders
../venv/bin/python src/boards.py renders renders renders
../venv/bin/python src/export_glb.py model/cloud_tide_pavilion.blend model/cloud_tide_pavilion.glb
```

调整造型时修改 `src/plan.py`：三片屋顶的 `rin_keys` / `w_keys` / `h_keys` 控制点，`SOLSTICE_SUNSET_AZ`（换纬度即可重算轴线），`BASIN_STEPS` 与潮位，坡道参数。木梁、幕墙高度、柱高、栏杆、分析图都会随之更新；改完运行 `python src/structure.py` 看支承距离、初步校核与抗侧力是否仍在合理范围，运行 `python src/program.py` 看面积与疏散距离。
