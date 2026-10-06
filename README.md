# 第五人格地图路线识别与投影工具 1.1

面向 Windows 的地图布局识别与透明路线投影工具。通过截图识别游戏地图布局，将预先制作的地图／路线图片覆盖到游戏地图上，并提供独立参考图、采集器、训练与资产维护工具。

当前项目版本为 **1.1**。版本来源为 `src/version.py`，主窗口标题、首页版本标签、命令行、Qt 应用信息和 Windows EXE 构建信息共用该版本。Windows 文件／产品版本的数值形式为 `1.1.0.0`，显示文本为 `1.1`。

1.1（2026-10-06）汇总自 1.0 发布以来的更新：正式接入图标与结构布局识别并设为新用户默认；地图与三套分层路线资源扩展至 `layout_01～layout_27`；模板模式提供展十、凉哈皮、原版干净底图和两套路线图底图版共五个显示版本，并完善版本切换、切层重匹配及异步图层提交。正式 CNN 模型仍覆盖 `layout_01～layout_21`，没有随此次发布重训；资源与配置检查不代替游戏内对齐验证。

项目总说明在本文，文件职责见 [项目目录与文件说明](项目目录与文件说明.md)，新增布局的完整操作见 [新增布局教程](新增布局教程.md)；[AGENTS.md](AGENTS.md) 保留为开发协作规则。本文以当前代码、配置与实际资源为准，不将历史实验结果作为准确率或性能保证。

## 目录

- [功能与支持范围](#功能与支持范围)
- [安装、启动与发行](#安装启动与发行)
- [日常使用](#日常使用)
- [配准与投影](#配准与投影)
- [设置、界面与状态保护](#设置界面与状态保护)
- [诊断与问题排查](#诊断与问题排查)
- [样本采集与模型训练](#样本采集与模型训练)
- [Demo 与资产维护](#demo-与资产维护)
- [评测与验证](#评测与验证)
- [目录边界与扩展](#目录边界与扩展)
- [新增布局完整教程](新增布局教程.md)

## 功能与支持范围

| 功能 | 当前行为 |
| --- | --- |
| 布局识别 | 可选 CNN 或图标与结构匹配，核对入口图后锁定正式布局并继续配准投影；匹配分数与 CNN 置信度分别显示 |
| 人工选择 | 可直接选择正式布局并确认，取消保留原锁定；选择“自动判断”解除锁定 |
| 定尺度模板匹配 | 默认方式，使用保存尺度匹配并投影三层完整原图，跳过楼层与 mylocation 检测 |
| 多尺度结构配准 | CPU 或 GPU 优先，搜索等比例缩放与平移，通过底图／截图证据检查后投影 |
| 路线版本 | 多尺度模式提供三个分层版本并复用有效姿态切换；模板模式提供展十、凉哈皮及各自底图版整图和原版干净底图拼接，共五个版本，切换后重新匹配 |
| 楼层管理 | 多尺度模式支持自动识别、截图前后复核、切层检测及手动选择；模板模式显示三层整图 |
| 原生覆盖层 | 透明、置顶、鼠标穿透、不抢游戏焦点，不透明度可调，地图关闭时隐藏 |
| 参考路线图 | 左上角独立参考窗，无需配准；多尺度模式需要有效布局和楼层，模板模式显示完整原图 |
| 快捷键与状态管理 | 地图开关、布局识别、重新投影、参考图显隐、Esc 关闭；前台窗口校验与迟到任务拦截 |
| 图形界面 | PyQt6 正式界面，背景图、可拖动标题栏、最小化／最大化、状态提示、截图与投影预览、运行日志 |
| 诊断 | 保存截图、掩膜、对齐预览、透明图层、合成图和 JSON，记录耗时、拒绝原因及后端回退 |
| 开发工具 | PyQt6 固定区域采集器、Tkinter Demo、CNN 训练、样本维护、离线标定、预览、尺度导入及评测 |
| Windows 发行 | PyInstaller 完整功能文件夹发行版，资源白名单收集，保留三种配准方式与模型 |

当前已标定资源为 **厄运之女／噩梦，layout_01～layout_27，共 27 个布局**。每布局包含 `2nd`、`1st`、`b1` 三层；三个分层显示版本各覆盖 81 个楼层。另有 27 张入口确认图、凉哈皮与展十各 27 张原始路线图。程序按资源目录发现布局，不在通用逻辑中固定数量。

当前定尺度模板配置覆盖 `layout_01～layout_27`，保留 81 条展十楼层记录；展十、凉哈皮及各自底图版均有 27 张原图，原版干净底图使用 27 组现有三层图片在内存中拼接。两种展十版本各有 27 个固定候选，两种凉哈皮版本及原版干净底图各有 81 个分层尺度候选。新增六个布局的展十保存尺度已按原图 SHA256 导入，凉哈皮尺度由现有标定离线推算；底图版逐层复核结构坐标后继承对应原图尺度。图标与结构识别动态读取二楼底图，正式 CNN 模型仍仅覆盖 `layout_01～layout_21`，CNN 识别新增布局需要发布包含新类别的模型。尺度配置及资源校验不等于游戏内对齐验证。

路线来自现成图片。项目没有自动寻路、角色控制或游戏内存读取功能。分类置信度和模板匹配分数都不是实际准确率。

## 安装、启动与发行

### 源码运行

目标环境：Windows 10/11、Python 3.10+、PowerShell。以下命令从项目根目录执行。已有 `.venv` 可直接使用；仅在环境缺失时创建。

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src --version
.\.venv\Scripts\python.exe -m src --check
.\.venv\Scripts\python.exe -m src
```

也可双击根目录 `启动路线投影程序.bat`。正式入口是 `python -m src`，同一路径下正常启动只允许一个实例；更新源码后应先关闭旧窗口再启动。

正式依赖包括 PyQt6、PyTorch、OpenCV、NumPy、Pillow、MSS、PyYAML 和 Windows 语音相关依赖。当前 `requirements.txt` **已经包含 PyTorch**；`requirements-training.txt` 复用正式依赖，`requirements-dev.txt` 额外安装 pytest。CPU 可独立完成配准，但完整程序仍需要 PyTorch 进行 CNN 识别。

CUDA 是否可用取决于显卡、驱动和已安装的 PyTorch 构建。GPU 配准还检查 OpenCV 插值兼容性；不可用时可用 CPU。配准后端选择不改变布局分类设备，当前正式应用加载 CNN 使用 CPU。

### 已构建 EXE

解压完整发行文件夹，双击 `IdentityVRouteAssistant.exe`。无需另装 Python，必须保留同目录资源与依赖，不能单独复制 EXE。将发行目录放在当前用户可写的位置，运行后会在其 `runtime/` 保存个人设置和诊断。

```powershell
.\IdentityVRouteAssistant.exe --version
.\IdentityVRouteAssistant.exe --check
```

更新时关闭旧程序；如需保留偏好，复制旧版 `runtime/settings.json` 到新发行目录。GPU 是否工作仍取决于目标机器环境。

### 构建 Windows EXE

```powershell
.\.venv\Scripts\python.exe -m pip install -r tools/release/requirements-build.txt
.\.venv\Scripts\python.exe tools/release/build_windows_exe.py --help
.\.venv\Scripts\python.exe tools/release/build_windows_exe.py
```

默认输出 `workspace/releases/windows_<时间>/IdentityVRouteAssistant/`。构建目录必须是新的工作区路径，已有产物不覆盖。脚本检查运行资源、标定状态和模板指纹，按白名单收集资源，不打包本机个人 runtime、训练样本或开发工具；构建完成后附带本文、目录说明和新增布局教程。

`build_info.json` 记录应用版本、Windows 数值版本、依赖版本和构建命令；`asset_manifest.json` 记录资源大小及 SHA256。EXE 属性中的文件／产品版本通过 `--version-file` 写入。既有发行文件夹及压缩包属于当时的构建产物，源码升级不会自动改写它们；发布 1.1 EXE 时需重新构建。普通打包不提供资产加密保护。

## 日常使用

1. 启动程序，在“更多设置”核对游戏窗口标题、截图区域、参考分辨率和游戏地图键。程序的地图键应与游戏一致。
2. 游戏打开地图后按 **F8** 识别布局，在入口确认框核对候选并锁定。正式候选无论置信度高低均允许人工确认；低置信度不会因开启自动锁定而被直接采用。
3. 也可在首页布局下拉框手动选择，再点击确认框的“确认使用”；取消恢复原选择。
4. 按游戏地图键开图并自动投影，默认 **G**；按 **F10** 手动重新投影。首页识别／手动投影按钮留出切回游戏的等待时间，执行时仍检查游戏前台状态。
5. 按 **H** 切换参考图显隐；首页也可控制投影图显隐和不透明度。参考图与已对齐投影是独立窗口，手动隐藏选择在本次地图内保留。
6. 按地图键关图或 **Esc**，投影立即隐藏。Esc 只有关闭语义，不用于开图。需要重新识别时再按 F8。

| 操作 | 默认键／入口 |
| --- | --- |
| 游戏地图开关 | G |
| 识别并确认布局 | F8 |
| 手动重新投影 | F10 |
| 左上角参考图显隐 | H |
| 关闭地图投影 | Esc |
| 地图状态与游戏不一致 | 首页“复位地图”，确认游戏地图已关闭后再开图 |

模板模式启动时不根据画面自动纠正开关状态。建议先关闭游戏地图，再点击“复位地图”，之后按地图键打开。游戏没有响应按键、使用其他方式关图等情况也可能需要复位。

## 配准与投影

### 布局识别方法

设置中的“布局识别方法”提供 `cnn`（CNN）和 `icon_structure`（图标与结构匹配，默认）。使用图标与结构匹配时，请在二楼地图按 F8：算法按截图区域的参考尺寸归一化，固定搜索 0°，通过入口图标锚定和墙线／地面评分给出正式 `layout_XX` 候选；图标不可靠时使用墙线搜索。

识别候选直接来自当前地图／难度的 `clean/<layout_id>/<layout_id>_2nd.png`，无须复制实验日期目录或训练模型。首次后台准备图标索引较慢，缓存按底图、图标、算法和 OpenCV 版本失效。结果显示匹配分数而非概率，始终人工核对前两名入口图并确认；确认后可切换楼层，正常使用定尺度、CPU 或 GPU 优先配准及参考路线。切换识别方法会清除旧锁定和结果，重新识别后再投影，不自动回退 CNN。

该算法已从独立实验副本移入正式入口，旧实验副本已于 2026-10-02 清除；本次离线和窗口验证不能代替游戏内实机检查，已知入口样例存在首选误识别，仍须核对候选。

### 定尺度模板匹配（默认）

`registration_backend=template` 使用已确认布局的保存尺度，匹配三层完整原图并直接投影。实际入口是 `src/registration/template_matching/direct.py`，尺度及源图指纹来自 `configs/template_matching.json`。

- 不执行楼层识别、mylocation 检测或多尺度结构验收，最高分位置直接作为候选显示。没有可匹配像素、尺度无效或资源指纹错误时停止本次投影。
- 坐标是 `original_source_image`，结果为 `floor=all`，`variant_id` 为实际所选的模板版本 ID；矩阵是 `screenshot_to_original` 和 `original_to_screen`，不能当作 `clean_to_screen`。
- 展十保留近黑背景透明处理；凉哈皮匹配前景为灰度大于 48 的像素，透明图层额外保留高饱和度彩色路线。超出截图区域的内容裁剪，原图可能含邻层文字或路线。
- 首页可选择展十、凉哈皮、原版干净底图，以及“凉哈皮单人速刷路线图底图版”和“展十速刷路线图底图版”，默认展十；楼层选择仍禁用。模板偏好保存到 `template_route_versions`，独立于 CPU/GPU 的 `route_versions`。
- 切换版本时隐藏旧图层并使旧任务失效。地图打开且游戏在前台时后台重新截图匹配，否则下次投影使用新版本；不会复用另一版本的原图矩阵。
- 凉哈皮部分原图三层比例不同，每层保存一个固定尺度并限制在对应原图区域内匹配，最高分候选用于整图投影；候选楼层仅是尺度来源，不是楼层识别结果。不进行连续尺度搜索，截图可见区域无法容纳时拒绝该候选，不暗中缩放。
- 点击游戏楼层按钮区域时使旧任务失效并隐藏图层，鼠标释放后等待截图延迟再匹配；不识别按钮高亮或编号。关闭自动投影后需要 F10 手动匹配。
- 楼层按钮点击采用约 25 ms 轮询，极短点击、键盘切层及滚轮缩放可能无法自动检测，应按 F10 或关图后再开图。
- 实时图层经 Qt 信号优先提交，然后后台合成预览及保存诊断；迟到图层和预览分别检查会话、代次和地图状态。

模板分数只说明候选相关性。稀疏地图结构可能与其他位置相似，需要核对实际重合效果；模板模式没有独立画面标识纠正地图开关状态。

原版干净底图按二楼、一楼、地下室原尺寸在内存中纵向拼接，不生成额外图片；每层使用离线推算的保存尺度及对应搜索区域，最高分候选用于整图投影，候选楼层不是独立楼层识别结果。图层与参考图均使用同一拼接坐标，切换版本必须重新匹配。

两套路线图底图版直接读取 `sources/route_lhp_solo_clean/` 与 `sources/route_zhanshi_fast_clean/`，ID 与目录名相同，保留图片中的路线和文字。它们与对应原图同名、同尺寸；导入工具逐层验证结构坐标一致后，继承原图的固定尺度和搜索区域，分别保存源图指纹，并验证原图、标定及参考版本记录，资源改变后须重新导入。这两个版本只在模板模式提供，原版干净底图仍使用三层底图拼接。

```powershell
.\.venv\Scripts\python.exe tools/assets/imports/import_template_source_variants.py
# 默认输出 workspace/asset_build/configs/template_matching_source_variants.json，核对后再复制到正式配置。
.\.venv\Scripts\python.exe -m src --project-once tests/fixtures/registration/game2.png --layout layout_05 --registration-backend template --route-version route_lhp_solo_clean --output workspace/reports/template_source_variants/lhp_clean
```

凉哈皮及原版干净底图的初始固定尺度由现有两套路线标定和展十保存尺度离线推算，尚不能视为游戏内对齐验证。`configs/template_matching.json` 保留旧 `entries`，新增 `original_variants` 保存版本原图／底图拼接记录、候选尺度、搜索区域及推算依赖指纹。底图、标定、原图或参考尺度变化后，须重新推算：

```powershell
.\.venv\Scripts\python.exe tools/assets/imports/derive_template_variant_scales.py
.\.venv\Scripts\python.exe tools/assets/imports/import_template_source_variants.py --config workspace/asset_build/configs/template_matching.json
# 检查最终输出 workspace/asset_build/configs/template_matching_source_variants.json，再复制到 configs/template_matching.json
```

`--check` 的 `template_originals` 列出各版本原图、固定候选数量和指纹错误；这些检查不代替实际图像对齐。

### 多尺度结构配准（CPU／GPU）

CPU/GPU 共用正式调度 `src/registration/multiscale_structure/runtime.py`，不能把旧 `src/registration/runtime_registration.py` 的入口视作默认主流程。

| 设置值 | 行为 |
| --- | --- |
| `cpu` | 强制原 CPU，GPU 模块延迟导入且此分支不依赖 CUDA |
| `auto` | GPU 优先；CUDA／插值不兼容、初始化、预热或运行异常时回退 CPU |
| `template` | 上述整张原图模板匹配，独立策略 |

GPU 使用 CUDA 颜色运算和稀疏评分，同时保留 CPU OpenCV 搜索、结构分析及几何检查。**正常配准拒绝不触发 CPU 重试**；GPU 故障后该进程保持 CPU 回退，重启后重新尝试。预热、模板缓存和并发访问有锁保护，CPU/GPU 动态缓存分离。

流程为：等待截图延迟 → 可靠楼层识别 → 地图截图 → 再次确认楼层 → 独立单帧配准 → 投影。楼层探测不使用 mylocation；失败可在约 500 ms 窗口内重试。开图楼层样本仅在同会话、同地图代次、同窗口位置／尺寸且不超过 100 ms 时复用，截图后仍复核；前后楼层不一致会拒绝。关闭自动楼层时可使用手动楼层，使用者需保证与游戏一致。

运行时只搜索等比例缩放和平移，结合结构、语义、边缘、已知背景冲突、空间竞争、可见区域与缩放稳定性验收。未知迷雾与已知背景分开处理，未知区域不作为“没有房间／走廊”的负证据。每帧独立，不读写可信历史；保留历史回归模块不意味着启用多帧证据。

单类别补充接受只在整片类别一致率轻微不足且其他检查全部通过时尝试，不能按通过检查数量抵消失败。当前下限为 0.68，常规门槛为 0.72；同时要求结构、可信／内部语义、结构线支持各至少 0.95、两个方向各至少 3 条线、线段中位残差不超过 1.5 屏幕像素、可见扰动 P95 不超过 4 像素、全视野位移 P95 不超过 12 像素、留一线段 P95 不超过 16 像素，并且所有竞争与其他必要门槛通过。指标用于稳定性评估，不是人工真值误差。

多尺度模式的路线图片墙线、覆盖率和路线补充确认不参与配准接受或拒绝。接受后显示完整路线，局部证据范围不限制最终显示范围；`clip_local_layer()` 保留用于离线诊断。

### 分层路线与标定关系

| 显示版本 | 持久化 ID | 内容 |
| --- | --- | --- |
| 原版干净底图 | `route_clean` | 引用底图，单位标定矩阵 |
| 凉哈皮单人速刷路线图 | `route_lhp_solo` | 预先制作并标定的透明分层路线 |
| 展十速刷路线图 | `route_zhanshi_fast` | 预先制作的透明分层路线及结构图 |

原版干净底图的深色背景在显示时透明化；分层素材若自带 alpha，仍保留原有透明度，源 PNG 不改写。

旧 `route_01` 设置与命令行作为 `route_lhp_solo` 的兼容别名。多尺度三版本使用同一底图配准结果：`asset_to_screen = clean_to_screen @ asset_to_clean`，矩阵计算前统一为齐次形式。离线标定可用 SIFT、RANSAC、仿射／单应矩阵及 ECC，不改变运行时搜索假设。

首页即时切换版本可复用仍有效的姿态，无需重做分类和搜索，但仍需加载有效新资产并核对布局、楼层、会话及地图状态。展十的 `*_structure.png` 是保留的资产校验／回归依赖，不应当作预览删除。资源路径保存为项目相对路径。

## 设置、界面与状态保护

默认配置来自 `configs/app_defaults.json`，个人设置在 `runtime/settings.json`；加载时补齐旧配置字段并兼容旧后端、路线 ID。新用户默认图标与结构识别、定尺度模板匹配，已有合法个人后端选择不覆盖。

| 设置 | 出厂值／说明 |
| --- | --- |
| 自动投影 | 开启，地图打开后自动投影 |
| 自动识别楼层 | 开启，仅多尺度模式使用 |
| 截图延迟 | 150 ms，可设 0～500 ms；控制开图首次等待和多尺度失败重试间隔 |
| 高置信度自动锁定 | 关闭；开启后仅置信度与分差均达标时跳过确认框 |
| 截图 ROI | `[901, 185, 1180, 1050]`，格式为左、上、宽、高，需按个人游戏画面核对 |
| 参考分辨率 | `[2560, 1440]`；按当前游戏客户区位置与尺寸换算 ROI，检查屏幕边界 |
| 游戏窗口标题 | `第五人格`、`Identity V`、`IdentityV`，按标题关键词识别前台 |
| 投影不透明度 | 34%，首页／设置页可调；“恢复默认投影”读取同一默认配置并恢复至 34% |
| 参考图不透明度 | 默认 47%，作为独立偏好保存 |
| 快捷键 | 点击设置页的键位按钮后按目标键，检查重复与合法性 |

设置页修改自动保存，保存失败会明确提示。分类门槛读取模型检查点，缺省置信度为 0.70、前两名差距为 0.20；正式流程没有 Demo 的探索比例门槛。

首页包含当前截图、匹配模板、重叠对比、最终投影预览、候选列表、布局／楼层／路线状态和耗时。日志可查看及导出，识别通知和确认窗便于核对结果。

地图键与 Esc 使用独立线程的 Windows Raw Input 队列，保持按键正常传给游戏，长按去重。短按按下／释放事件可在 GUI 忙碌期间缓存；关闭事件优先于结果应用。设置页暂停地图操作，前台游戏校验继续生效；监听失败会隐藏投影并提示重启。

识别、截图、配准、预热及诊断写入在后台执行，Qt 控件在主线程更新。关闭地图、切层、换布局、换后端、复位或会话失效会使旧任务失效；快速关图再开图只保留最新有效请求，迟到任务不能恢复旧图层、确认框、参考图或证据。

## 诊断与问题排查

- 正式日志在 `runtime/logs/`，启动异常写入 `app_error.log`。查看按键事件序号、前台标题、处理等待、地图代次及丢弃原因可定位开关状态问题。
- 投影诊断在 `runtime/diagnostics/route_projection/app_runs/`，每次运行保存 `01_screenshot.png`、可用的观测／掩膜／对齐图、`05_projection.png`、`06_overlay_layer.png` 和 `result.json`，另有楼层与迷雾诊断。
- 地图探测记录在 `runtime/diagnostics/map_presence/`。当前 CPU/GPU 使用楼层记录，旧 mylocation 字段或图片只作兼容，不能据旧文件判断本轮进行了该识别。
- 地图探测 JSON 分段记录首次等待、线程等待、窗口查询、ROI 换算、截图转换、楼层识别、回传等待、主线程应用及重试等待。后台诊断为串行写入，只保留最新待写任务，极频繁操作可能合并中间诊断。
- 图层提交耗时是主线程提交耗时，**不是物理屏幕显示延迟**；预览合成与保存耗时单独记录。设定截图延迟也不是硬实时截图时刻，仍受系统调度影响。

| 现象 | 处理 |
| --- | --- |
| 地图状态与游戏不一致 | 先实际关闭游戏地图，点击“复位地图”，再按地图键打开 |
| F8／F10 没有按预期执行 | 检查游戏是否前台、窗口标题和快捷键是否一致，关闭设置页后重试 |
| 模板错位或切层后未更新 | 核对布局、游戏缩放、保存尺度与截图区域，按 F10 重配；查看候选提示 |
| 多尺度拒绝 | 查看 `result.json` 的拒绝原因，不要将拒绝改成成功；楼层不一致或证据不足应重新截图 |
| GPU 回退 CPU | 查看实际后端、设备和 fallback_reason；重启后可重新尝试 GPU |
| CNN 无模型或新布局识别不到 | 检查模型路径、标签及资源是否配套，增加图片目录不等于支持新类别 |
| 资源指纹不一致 | 核对源图与保存尺度／标定，重新核验后更新相应配置和缓存 |
| Windows 捕获排除失败 | 采集器取消截图或隐藏边框以避免采入自身，检查系统能力与界面提示 |

## 样本采集与模型训练

### 固定区域采集器

```powershell
.\.venv\Scripts\python.exe tools/collector/run_fixed_roi_collector.py --check
.\.venv\Scripts\python.exe tools/collector/run_fixed_roi_collector.py
```

也可双击 `tools/launchers/启动固定区域样本采集器.bat`。采集器使用 PyQt6，截图和 PNG 保存串行在后台完成；支持快照框选／物理坐标输入、负坐标显示器、预览适应窗口／1:1／缩放／拖动、截图边框、中文语音提示、精确重复图跳过、启动最小化和采集间隔。

默认间隔 0.1 秒，默认启用重复跳过及语音，关闭边框和启动最小化。区域及个人选项保存到 `workspace/tool_state/capture.json`；默认值在 `tools/configs/capture_defaults.json`，兼容旧路径。选择新区域自动保存，其他选项点击保存。

| 采集操作 | 默认快捷键 |
| --- | --- |
| 只预览 | F5 |
| 采集一张 | F10 |
| 开始连续采集 | F6 |
| 暂停／继续 | F7 |
| 停止 | F8 |
| 归档为下一个新布局 | Ctrl+Alt+N |
| 补充到明确选择的已有布局 | Ctrl+Alt+E |
| 重建 JSONL 索引 | Ctrl+Alt+R |

采集先写 pending，编号根据现存已归档样本分配，不根据地图资产数量推断。暂停／停止／关闭等待当前文件操作完成，不再发起下一张。未归档样本可在下次继续归档；手动删掉的图片不会被归档。最近列表最多 60 张，“清空”只清显示记录。

右键可永久删除样本，同时更新 JSONL，不保留删除副本。命令行入口：

```powershell
.\.venv\Scripts\python.exe tools/training/delete_sample.py --image "workspace/datasets/layout_classification/images/地图/难度/layout_01/样本.png"
.\.venv\Scripts\python.exe tools/training/rebuild_layout_jsonl.py
```

采集器主窗、区域选择窗、菜单、提示及边框默认使用 Windows 捕获排除，并在句柄变化或截图前核对；失败时阻止污染样本。该能力要求 Windows 10 2004+ 和桌面合成，取决于捕获接口，不能保证所有第三方录屏方式遵守。已保存／删除／跳过统计是当前上下文的会话统计，不是全数据集总数。

### CNN 训练与候选模型

训练前重建 JSONL，确认标签、空编号和采集质量。脚本按现存 JSONL 发现布局；每布局使用后段样本进行时间顺序验证，默认排除开局 12% 公共画面，验证比例 20%。

```powershell
.\.venv\Scripts\python.exe tools/training/train_layout_classifier.py --help
.\.venv\Scripts\python.exe tools/training/rebuild_layout_jsonl.py
.\.venv\Scripts\python.exe tools/training/train_layout_classifier.py --preprocess original --device cpu
.\.venv\Scripts\python.exe tools/training/train_layout_classifier.py --preprocess pre_resized --device cuda --batch-size 128 --workers 4
```

上面的 CUDA 命令仅在已有 CUDA 环境时使用。实际默认训练参数为 24 epochs、batch-size 32、lr 0.001、seed 20260918；设备自动选择可用 CUDA 或 CPU，workers 默认分别为 4 或 0。

| 模式 | 预处理与输出 |
| --- | --- |
| `original` | 原始图增强后缩放；输出 `layout_classifier.pt` 与 `layout_classifier_validation.pt` |
| `pre_resized` | 缓存 160×144 无损缩略图后增强；输出对应 `_pre_resized.pt` 与 `_pre_resized_validation.pt` |

检查点在 `workspace/checkpoints/`，报告在 `workspace/reports/classification/`，包括准确率、逐布局结果和混淆矩阵。两种模式文件互不覆盖；预缩放缓存根据源路径、大小及修改时间失效。保存最佳验证模型后另做全量样本微调。

`--max-layout N` 可只训练编号范围；`--allow-missing-layouts` 允许跳过范围内空类别而不重编号。训练不会自动发布至 `models/releases/`。模型更新需要单独核对标签、预处理、独立验证及可回退文件，再替换正式模型和必要配置。

## Demo 与资产维护

### 两个 Tkinter Demo

```powershell
.\.venv\Scripts\python.exe tools/demos/run_layout_recognition_demo.py
.\.venv\Scripts\python.exe tools/demos/run_layout_recognition_demo.py --model workspace/checkpoints/layout_classifier.pt
.\.venv\Scripts\python.exe tools/demos/run_route_projection_demo.py --backend cpu --check
.\.venv\Scripts\python.exe tools/demos/run_route_projection_demo.py --backend auto
```

双击入口在 `tools/launchers/`。布局 Demo 默认 F9 截图，展示全部置信度排序，支持已有原图、预缩放及自选模型、刷新模型列表和类别覆盖提示；保留探索比例门槛，与正式确认策略不同。路线 Demo 默认 F9 识别、F10 投影，手动选楼层，调用正式单帧多尺度 workflow，支持 `cpu|auto` 和离线 `--project-once`。

Demo 设置在 `workspace/tool_state/`，缓存、报告在 `workspace/cache/` 与 `workspace/reports/`；不写正式个人设置、不读取可信历史。Demo 的 Tkinter 覆盖窗与正式 Qt 窗口分离。

### 资产入口

| 路径（相对于 `tools/assets/`） | 功能 |
| --- | --- |
| `calibration/calibrate_route_assets_batch.py` | 凉哈皮原图分层、离线标定、透明路线图与摘要 |
| `imports/derive_template_variant_scales.py` | 从现有标定推算凉哈皮与原版干净底图各楼层固定尺度，输出暂存配置 |
| `imports/import_template_source_variants.py` | 逐层验证两套路线图底图版坐标一致，继承对应原图固定尺度并生成暂存配置 |
| `calibration/calibrate_zhanshi_routes.py` | 展十独立楼层定位、精化、透明图和结构图，支持选布局及 render-only |
| `calibration/organize_map_assets.py` | 底图单位矩阵版本、资源清单与目录索引 |
| `previews/export_all_route_registration_previews.py` | 将各版本路线按标定叠到底图并导出预览 |
| `previews/export_three_version_overlaps.py` | 三版本两两与整体叠加、离线对照画廊 |
| `previews/create_calibration_contact_sheet.py` | 从标定摘要生成拼图 |
| `previews/build_route_review.py` | 从资产审计、配准回归和定尺度固定投影产物生成独立人工复核网页，支持叠加、分割、放大及结论 JSON 导入导出 |
| `previews/project_route_asset.py` | 使用已有底图矩阵投影指定资产，不做配准搜索；拒绝原图坐标冒充底图坐标 |
| `imports/import_template_scales.py` | 从外部 scale_config.json、config.json、images 导入保存尺度并核验内容指纹 |
| `cache/build_local_structure_library.py` | 批量构建底图局部结构索引和特征预览 |

所有入口支持 `--help`。标定／整理工具默认写 `workspace/asset_build/`，仅显式 `--publish` 才更新正式资产；模板尺度导入默认生成暂存配置。预览与审计默认写工作区。

人工复核网页使用 `tools/assets/previews/build_route_review.py --report <含资产审计与两套定尺度固定投影的报告目录> --output workspace/reports/manual_route_review` 生成，当前报告目录为 `workspace/reports/template_source_variants_20261003`。用浏览器打开输出的 `index.html` 即可离线比较并记录结论；图片已内嵌，无须部署网站。记录保存在浏览器，也可导出 JSON；若内置浏览器未下载，可复制页面中的 JSON 保存。此工具会重放三个配准回归样例，不修改正式标定与程序设置。

```powershell
.\.venv\Scripts\python.exe tools/assets/calibration/calibrate_route_assets_batch.py --help
.\.venv\Scripts\python.exe tools/assets/previews/export_three_version_overlaps.py --output workspace/reports/three_version_overlaps
.\.venv\Scripts\python.exe tools/assets/imports/import_template_scales.py --source "外部尺度目录"
```

正式地图路径为 `assets/maps/<map_id>/<difficulty_id>/`，其中 `clean/<layout_id>/` 是底图，`entrance/` 是入口图，`sources/<variant_id>/` 是原图，`routes/<variant_id>/<layout_id>/` 是运行图层和 `calibration.json`。替换图片或特征逻辑后核验对应标定与指纹，不能跨布局、楼层或源图错误复用缓存。

重新制作 `clean` 后，重新标定对应布局的凉哈皮、展十分层图及结构图，并同步 `configs/template_matching.json` 的底图指纹、标定指纹和 `clean_to_original`；展十完整原图未变时保留其保存尺度，再重新推算 `original_variants` 的凉哈皮及原版干净底图尺度及依赖指纹，随后用 `import_template_source_variants.py` 重新核对并导入两套路线图底图版。原版底图继续使用单位矩阵引用。更新后重启程序，加载新资源并重新准备模板缓存。

## 评测与验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src --check
.\.venv\Scripts\python.exe -m src --project-once tests/fixtures/registration/game2.png --layout layout_05 --floor 2nd --route-version route_lhp_solo --registration-backend cpu --output workspace/reports/manual_check
```

`--check` 汇总模型是否存在、布局和各版本楼层状态；退出码 0 不代表质量已验证，仍须看缺失模型及 REVIEW。离线投影拒绝时退出码 2，输出诊断解释原因。`--smoke` 启动正式窗口后约 2 秒关闭，用于界面烟雾检查，不是游戏内验证。

1.1 发布检查（2026-10-06）：模型、27 个布局、三套分层版本和五种模板版本的资源及指纹检查通过；分层资产审计覆盖 81 个楼层、243 份版本记录，无加载错误，14 项结构指标需人工复核。固定截图 `game2.png` 的 `layout_05` 凉哈皮底图版模板投影通过，并检查了合成预览。本次没有重新构建 EXE，也未做游戏内实机检查；这些离线结果不代表全部布局的实机对齐已验证。

本次完整测试仍有三项配准回归失败，尚未完成原因定位：`test_first_sparse_registration.py` 的 layout_11 地下室稀疏候选冲突样例被拒绝；`test_joint_acceptance.py` 的 layout_16 地下室已复核房间样例未获补充接受；`test_pose_evidence.py` 的 layout_13 地下室区域竞争样例未按预期拒绝。测试断言和配准验收策略保持原样；不能将本次发布描述为全部回归通过。复现可运行这三个测试文件，人工复核工具为 `tools/assets/previews/build_route_review.py`。

| 六个入口（`tools/evaluation/`） | 作用 |
| --- | --- |
| `audit_route_assets.py` | 检查底图、路线和标定质量，REVIEW 需要人工核对 |
| `audit_registration_runs.py` | 按 `--day YYYYMMDD` 汇总运行截图及人工复核队列 |
| `audit_pose_quality.py` | 比较已知姿态合成报告并生成未标注实图复核队列 |
| `benchmark_first_registration.py` | 当前独立单帧真实截图回放，支持显式参考报告 |
| `benchmark_first_synthetic.py` | 固定随机序列的已知变换，narrow／partial／spread 遮挡测试 |
| `verify_gpu_production.py` | 固定样例真实 CPU/GPU 运行与路线版本投影一致性，需要实际 CUDA |

```powershell
.\.venv\Scripts\python.exe tools/evaluation/audit_route_assets.py
.\.venv\Scripts\python.exe tools/evaluation/benchmark_first_registration.py --help
.\.venv\Scripts\python.exe tools/evaluation/benchmark_first_synthetic.py --backend cpu --limit 1 --output workspace/reports/synthetic_cpu.json
.\.venv\Scripts\python.exe tools/evaluation/verify_gpu_production.py --help
```

两个 benchmark 默认 CPU，`--backend auto` 为 GPU 优先；报告记录实际后端及回退原因，不使用可信历史。旧 baseline/optimized、搜索策略与 config 开关会提示迁移并退出，不可据它们切换正式算法。

真实回放清单是 JSON 数组，每项包含唯一 `source`、`layout`、`floor`、PNG 的 `sha256`；可用 `image` 指定项目相对或绝对路径。未指定 image 时读取 runs 下 `<source>/01_screenshot.png`，未指定 manifest 时从实际诊断取样。split 筛选需要对应字段；参考报告只在显式 `--reference` 时读取，需与筛选后的样例及截图／底图指纹完全对应。`--save-differences` 需配合 reference。

合成 `--limit` 限制布局／楼层对，每对三种遮挡；`--case layout_01/b1/narrow` 可选单例。报告使用 `rows[].outcomes[cpu|auto]`。耗时 P50/P95 包含接受和拒绝，接受无有效姿态单列，不将它们当作零误差。空输入或缺失样例是失败，旧报告不是本次验证证据。

比较性能需固定输入、线程、配置及历史状态，区分冷启动、模板准备和搜索。CPU 回退不能当作 GPU 验证；旧算法位置差异不是人工真值误差；已知变换和 Qt 模拟测试不等于实机准确率或游戏开图速度。

## 目录边界与扩展

| 目录 | 职责 |
| --- | --- |
| `src/` | 正式程序与公共算法，不依赖 tools、workspace、tests 或 Tkinter |
| `configs/` | 默认设置、地图目录、模板尺度与标定关联 |
| `assets/`、`models/releases/` | 正式图片、逐布局标定和发布分类模型 |
| `runtime/` | 本机个人设置、正式缓存、日志及诊断，默认不提交 |
| `tools/` | 采集、Demo、训练、资产、评测和 EXE 构建 |
| `workspace/` | 样本、索引、工具状态及按需生成的缓存、检查点、报告、暂存和发行产物 |
| `tests/` | pytest 回归与少量固定复现输入 |

各源码／脚本／配置／测试文件职责及重复资产规则见 [项目目录与文件说明](项目目录与文件说明.md)。`.git`、`.venv` 和缓存只说明用途，不展开依赖内部文件。第三方许可证和依赖文档保持原样，历史发行包内部说明属于原构建产物。

新增布局须配套三层底图、入口图与路线标定；模板模式另需完整原图及保存尺度。图标与结构识别动态读取正式二楼底图；使用 CNN 时还须配套采集样本和模型类别。持久化布局、楼层、版本 ID 保持兼容，资源路径写项目相对路径。训练、发布模型、正式资产更新是不同步骤。

逐步操作见 [新增布局教程](新增布局教程.md)：分别说明 `template`、`cpu`、`auto` 三种配准，以及三个分层路线版本，包含暂存发布、模板尺度导入和验证。CPU／GPU 共用分层路线；模板模式可选择展十、凉哈皮及各自底图版完整原图，以及三层原版干净底图的内存拼接版本。

已删除的 archives、旧 packaging、comparison、experiments 及历史维护报告不是当前入口或回退来源。开发工具可重新生成工作区目录；存在生成目录不代表历史备份恢复。当前 CPU 回退依赖保留的 CPU 源码。
