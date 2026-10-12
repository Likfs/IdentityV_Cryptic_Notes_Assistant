# 第五人格地图路线识别与投影工具 1.1

Windows 地图布局识别与路线投影工具。正式流程为：**游戏截图 → 布局识别（纯结构或图标与结构） → 人工确认布局 → 定尺度模板匹配 → 三层完整原图透明投影**。

CNN、CPU/GPU 多尺度结构配准、训练采集工具和专属测试已移除，旧 `retired_features/` 归档目录也已删除。需要查阅曾经提交的旧文件时，请查看 Git 历史；未提交的本机资料不在 Git 历史中。版本来源仍为 `src/version.py`，更新源码不会自动生成或改写 EXE。

## 功能与支持范围

- 纯结构识别默认使用 CPU（最多 8 个工作线程），要求游戏地图缩到最小，检索全部布局的三层底图；不依赖入口定位，不进行姿态精修。
- 可选图标与结构识别动态读取当前地图／难度的二楼底图，给出候选与入口确认图，无需训练模型。固定 0°，图标不足时使用墙线搜索。
- 定尺度模板匹配仅比较保存尺度候选；不识别楼层、不执行多尺度结构验收，最高分候选直接显示，并提示人工核对。
- 保留展十、凉哈皮、原版干净底图、凉哈皮路线图底图版、展十路线图底图版五种显示版本。
- 保留参考图、快捷键、可配置截图延迟、日志与诊断、版本切换和地图拖动跟随（试用）。不控制游戏角色，不读取游戏内存，不提供自动寻路。
- 当前资源覆盖厄运之女／噩梦 layout_01～layout_27，三层底图及保存尺度以 `--check` 实际输出为准。资源齐全、离线检查或合成样例通过不等于游戏内对齐准确率。

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

正式依赖为 PyQt6、OpenCV、NumPy、Pillow、MSS 及 Windows comtypes。运行不再依赖 PyTorch、CUDA 或 CNN 权重。已有开发环境可继续使用，无需重建；环境中以前安装的包不会自动卸载。

### 已构建 EXE

解压完整发行文件夹，双击 `IdentityVRouteAssistant.exe`。无需另装 Python，必须保留同目录资源与依赖，不能单独复制 EXE。将发行目录放在当前用户可写的位置，运行后会在其 `runtime/` 保存个人设置和诊断。

```powershell
.\IdentityVRouteAssistant.exe --version
.\IdentityVRouteAssistant.exe --check
```

更新时关闭旧程序；如需保留偏好，复制旧版 `runtime/settings.json` 到新发行目录。旧无效识别偏好会归一为纯结构识别，投影仍为定尺度模板匹配；已有图标与结构识别选择保留。

### 构建 Windows EXE

```powershell
.\.venv\Scripts\python.exe -m pip install -r tools/release/requirements-build.txt
.\.venv\Scripts\python.exe tools/release/build_windows_exe.py --help
.\.venv\Scripts\python.exe tools/release/build_windows_exe.py
```

默认输出 `workspace/releases/windows_<时间>/IdentityVRouteAssistant/`。构建目录必须是新的工作区路径，已有产物不覆盖。脚本检查运行资源、标定状态和模板指纹，按白名单收集资源，不打包 PyTorch、模型、retired_features、本机个人 runtime、训练样本或开发工具，也不打包旧分层投影 PNG；构建完成后附带本文、目录说明和新增布局教程。

`build_info.json` 记录应用版本、Windows 数值版本、Python／PyInstaller 版本、识别方式和构建命令；`asset_manifest.json` 记录资源大小及 SHA256。EXE 属性中的文件／产品版本通过 `--version-file` 写入。既有发行文件夹及压缩包属于当时的构建产物，源码升级不会自动改写它们；发布 1.1 EXE 时需重新构建。普通打包不提供资产加密保护。

## 日常使用

1. 启动程序，在“更多设置”核对游戏窗口标题、截图区域、参考分辨率和游戏地图键。程序的地图键应与游戏一致。
2. 游戏打开地图后按 **F8** 识别布局，在入口确认框核对候选并锁定。纯结构识别请将游戏地图缩到最小；图标与结构识别请在二楼地图操作。结果是匹配分数，必须人工核对；不能视为正确概率。
3. 也可在首页布局下拉框手动选择，再点击确认框的“确认使用”；取消恢复原选择。
4. 按下并松开游戏地图键开图并自动投影，默认 **G**。程序从松开地图键开始等待设定的截图延迟，按住期间不启动开图截图；按 **F10** 手动重新投影。首页识别／手动投影按钮留出切回游戏的等待时间，执行时仍检查游戏前台状态。
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

### 纯结构识别（最小缩放·CPU）

在“更多设置 → 布局识别方法”选择。新默认值为 `pure_structure`，已有合法的 `icon_structure` 个人偏好保留。仅支持最小缩放，不检测游戏缩放档位；其他缩放不能作为本模式的有效输入。

读取所有布局的三层底图，固定方向 0°，在归一化截图到原底图的 `1.46666667`、`1.51666667` 两个尺度上搜索。它们不是游戏百分比。保留墙线支持与冲突扣分两路候选，在粗搜位置作一次全分辨率评分，再按原图边缘和可见区域的局部缺线、多余墙线扣分排序。未知迷雾不作负证据；没有自动尺度搜索、尺度／位置精修或 GPU 路径。截图尺寸、ROI 或底图变化后需要重新验证适用性。

评分不等于正确概率，必须人工确认；候选楼层和姿态只供排序诊断，不用于投影。首次后台准备底图索引耗时与后续单张识别分开。索引按图片内容、路径、算法参数和 OpenCV 版本更新，CPU 线程池共享只读资源并按原输入顺序汇总。切换方法清除旧候选与布局锁，迟到结果不会恢复旧状态；关闭时后台释放线程。正式运行不依赖开发报告、截图清单或实验脚本。

### 图标与结构识别

候选来自二楼底图与入口图标；索引按底图、图标、算法及 OpenCV 版本失效。所有候选必须人工核对后锁定，布局确认后可以切层并重新匹配。

### 定尺度模板匹配

`registration_backend=template` 使用已确认布局的保存尺度，匹配三层完整原图并直接投影。实际入口是 `src/registration/template_matching/direct.py`，尺度及源图指纹来自 `configs/template_matching.json`。

- 不执行楼层识别、mylocation 检测或多尺度结构验收，最高分位置直接作为候选显示。没有可匹配像素、尺度无效或资源指纹错误时停止本次投影。
- 坐标是 `original_source_image`，结果为 `floor=all`，`variant_id` 为实际所选的模板版本 ID；矩阵是 `screenshot_to_original` 和 `original_to_screen`，不能当作 `clean_to_screen`。
- 展十保留近黑背景透明处理；凉哈皮匹配前景为灰度大于 48 的像素，透明图层额外保留高饱和度彩色路线。超出截图区域的内容裁剪，原图可能含邻层文字或路线。
- 首页可选择展十、凉哈皮、原版干净底图，以及“凉哈皮单人速刷路线图底图版”和“展十速刷路线图底图版”，默认展十；楼层选择仍禁用。模板偏好保存到 `template_route_versions`，独立于仅作旧设置迁移和离线维护使用的 `route_versions`。
- 切换版本时隐藏旧图层并使旧任务失效。地图打开且游戏在前台时后台重新截图匹配，否则下次投影使用新版本；不会复用另一版本的原图矩阵。
- 凉哈皮部分原图三层比例不同，每层保存一个固定尺度并限制在对应原图区域内匹配，最高分候选用于整图投影；候选楼层仅是尺度来源，不是楼层识别结果。不进行连续尺度搜索，截图可见区域无法容纳时拒绝该候选，不暗中缩放。
- 点击游戏楼层按钮区域时使旧任务失效并隐藏图层，鼠标释放后等待截图延迟再匹配；不识别按钮高亮或编号。关闭自动投影后需要 F10 手动匹配。
- 楼层按钮点击采用约 25 ms 轮询，极短点击及键盘切层可能无法自动检测，应回到保存尺度对应的开图缩放后按 F10 或关图重开。
- 实时图层经 Qt 信号优先提交，然后后台合成预览及保存诊断；迟到图层和预览分别检查会话、代次和地图状态。
- 投影覆盖窗口在隐藏前清空旧路线并立即重绘，重新显示前先完成窗口属性设置，减少旧画面在首次绘制前残留的可能；参考图保留原有隐藏行为。用户已在本次实机复测中确认移动后开图的闪烁消失，其他设备与场景仍需分别验证。
- “更多设置 → 地图拖动跟随（试用）”默认开启，说明为“拖动地图时路线同步平移（不支持缩放）”。沿用 `template_motion_tracking=true` 设置键以兼容已有偏好，关闭后恢复静态投影。仅在定尺度首次配准显示后启动；跟随仅按鼠标位移平移，不做自动图像校正。首次位置需要人工核对，不修改保存尺度、首次配准结果。
- 首次显示后后台准备完整原图的固定尺度缓存。覆盖窗口固定在地图截图区域，内部路线按鼠标物理像素位移平移并裁剪；原先在视野外的路线可随拖动进入画面。Windows 鼠标钩子只保存输入，Qt 主线程按 16 ms 定时器消费最新坐标（不代表实际帧率或显示延迟），拖动期间不截图、配准或重新缩放原图。静止时不更新路线，不会因地图结构稀疏而隐藏。
- 仅游戏在前台、投影显示且左键按下点位于地图区域并避开已知楼层按钮时开始跟随。每次拖动从按下时的坐标和累计偏移计算，松手补交最终坐标；保留短拖动的按下／松开顺序，拖动开始后允许鼠标移出区域。鼠标事件继续传递给游戏，投影保持鼠标穿透和不抢焦点。
- 暂不支持缩放。检测到地图区域滚轮操作后隐藏路线并结束跟随，需恢复保存尺度对应的开图缩放后按 F10 或关图重开；不会自动重新配准或推测滚轮倍率。其他方式的缩放暂不能自动检测。地图边界、游戏拖动倍率或动画若导致实际平移不等于鼠标位移，仍可能错位，需手动重新投影。
- 关图、切层、切布局／版本、进入设置、手动隐藏以及游戏切出前台或窗口位置／尺寸变化时，结束当前跟随并丢弃旧输入与迟到缓存。重新投影建立新会话；退出时卸载鼠标钩子。跟随不依赖截图排除接口是否成功，因为不采集后续图像。此功能尚待实机验证，初始配准诊断与预览仅代表首次截图。

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

## 设置、界面与状态保护

默认配置来自 `configs/app_defaults.json`，个人设置在 `runtime/settings.json`；加载时补齐旧配置字段并兼容旧后端、路线 ID。旧无效识别选择归一为纯结构识别，合法图标识别选择保留；投影后端仍固定为定尺度模板匹配；截图、热键、不透明度、模板版本和拖动跟随偏好继续保留。读取设置不主动改写个人文件，下一次保存写入当前有效选项。

| 设置 | 出厂值／说明 |
| --- | --- |
| 自动投影 | 开启，地图打开后自动投影 |
| 截图延迟 | 150 ms，可设 0～500 ms；从松开地图键起算开图首次等待 |
| 截图 ROI | `[901, 185, 1180, 1050]`，格式为左、上、宽、高，需按个人游戏画面核对 |
| 参考分辨率 | `[2560, 1440]`；按当前游戏客户区位置与尺寸换算 ROI，检查屏幕边界 |
| 游戏窗口标题 | `第五人格`、`Identity V`、`IdentityV`，按标题关键词识别前台 |
| 投影不透明度 | 34%，首页／设置页可调；“恢复默认投影”读取同一默认配置并恢复至 34% |
| 参考图不透明度 | 默认 47%，作为独立偏好保存 |
| 快捷键 | 点击设置页的键位按钮后按目标键，检查重复与合法性 |

设置页修改自动保存，保存失败会明确提示。布局始终人工确认。识别方法可选纯结构识别和图标与结构识别，配准方式仍固定为定尺度模板匹配；不提供已移除算法。

首页包含当前截图、匹配模板、重叠对比、最终投影预览、候选列表、布局／楼层／路线状态和耗时。日志可查看及导出，识别通知和确认窗便于核对结果。

日志控件只保留最近 2,000 行，新增内容按约 100 ms 间隔批量追加，切换等级筛选时才重建显示。窗口隐藏、最小化、等待开图截图或任务忙碌时暂缓日志绘制，恢复后补齐当前筛选的最新记录。导出仍包含本次启动后尚未清空的全部日志；清空会清除这部分可导出记录。

地图键与 Esc 使用独立线程的 Windows Raw Input 队列，保持按键正常传给游戏，长按去重。开图须先在游戏中按下，再在同一窗口松开；截图延迟从松开事件的检测时间起算，GUI 忙碌时仅等待剩余时间，不把按住时长算入截图等待。短按按下／释放事件可在 GUI 忙碌期间缓存。关图按下时即提前隐藏投影，随后松开不会重新开图；Esc 保持按下即关闭，并取消尚未松开的开图动作。关闭事件优先于结果应用。设置页暂停地图操作，前台游戏校验继续生效；监听失败会隐藏投影并提示重启。

识别、截图、配准及诊断写入在后台执行，Qt 控件在主线程更新。关闭地图、切层、换布局、复位或会话失效会使旧任务失效；快速关图再开图只保留最新有效请求，迟到任务不能恢复旧图层、确认框、参考图或证据。

## 诊断与问题排查

- GUI 运行日志保存在内存中，点击“导出”时默认保存到 `runtime/logs/`；未导出的历史日志无法在退出后还原。启动异常写入 `app_error.log`。查看按键事件序号、前台标题、处理等待、地图代次及丢弃原因可定位开关状态问题。
- 投影诊断在 `runtime/diagnostics/route_projection/app_runs/`，每次运行保存 `01_screenshot.png`、`05_projection.png`、`06_overlay_layer.png` 和 `result.json`；当前完整原图流程只生成截图、投影预览、透明图层和结果 JSON，不生成旧观测／楼层／迷雾诊断。
- 图层提交耗时是主线程提交耗时，**不是物理屏幕显示延迟**；预览合成与保存耗时单独记录。设定截图延迟也不是硬实时截图时刻，仍受系统调度影响。
- 模板投影在图层提交后立即显示完整计时，不等待后台预览。`timings_ms.total` 是截图之后的 workflow 处理耗时，不含开图等待、截图或 GUI 交付。`key_to_request` 为开图松键检测至请求开始（包含设定等待与可能的旧任务等待，不含按住时长），`request_setup` 为请求在主线程的准备时间，`worker_start_delay` 为线程启动等待；`result_queue` 为信号发出至接收函数入口，`result_input_handling` 为入口按键处理，原 `result_delivery` 保留两者之和。`key_to_overlay_submit` 记录松键至覆盖层提交；楼层按钮触发时两项计时起点为按钮释放，手动请求可能没有这些项。
- `result.json` 的 `gui_diagnostics.at_request` / `at_delivery` 保存请求开始／结果接收时的日志条数、启动以来累计条数和待刷新状态，以及日志刷新、状态更新、预览控件更新的累计次数、总耗时、最大和最近耗时（毫秒）。这些是主线程函数耗时，不是物理绘制耗时；旧诊断没有这些字段，不能反推出旧运行的日志数量。

| 现象 | 处理 |
| --- | --- |
| 地图状态与游戏不一致 | 先实际关闭游戏地图，点击“复位地图”，再按地图键打开 |
| F8／F10 没有按预期执行 | 检查游戏是否前台、窗口标题和快捷键是否一致，关闭设置页后重试 |
| 模板错位或切层后未更新 | 核对布局、游戏缩放、保存尺度与截图区域，按 F10 重配；查看候选提示 |
| 资源指纹不一致 | 核对源图与保存尺度／标定，重新核验后更新相应配置和缓存 |

## 资产维护与新增布局

正式素材在 `assets/maps/<map_id>/<difficulty_id>/`。`clean/` 为三层底图，`entrance/` 为人工确认图，`sources/` 为完整原图，`routes/` 为标定和离线维护资产。分层 PNG 不再用于正式投影，但仍供素材标定与对齐预览使用，不能随意删除标定指纹依赖。

保留的工具按 `tools/assets/calibration/`、`imports/`、`previews/` 分类，通用标定逻辑在 `tools/common/route_calibration.py`，默认输出至 `workspace/asset_build/` 或 `workspace/reports/`。尺度导入先生成暂存配置；只有明确发布任务才更新正式素材。详细步骤见 [新增布局教程](新增布局教程.md)。无需采集分类数据或训练 CNN。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src --check
.\.venv\Scripts\python.exe -m src --smoke
.\.venv\Scripts\python.exe -m src --project-once tests/fixtures/registration/game2.png --layout layout_05 --route-version route_lhp_solo --output workspace/reports/template_manual_check
```

临时目录权限受限时，pytest 可加 `--basetemp=workspace/tmp/<新的测试目录名>`；不要指向已有重要文件的目录。`--check` 检查资源与指纹，重点查看各版本 errors；它不验证游戏内对齐。离线投影需通过 `--layout` 指定已确认布局，未接受时退出码为 2。`--smoke` 仅验证界面启动和关闭。

正式程序不依赖 `tools/`、`tests/`、`workspace/`、`retired_features/` 或 Tkinter。默认 pytest 仅收集 `tests/`；旧功能专属测试已移除，不能把它们的历史通过记录算作当前版本验证。

测试按职责分组，默认全量运行单元、集成、资源和维护工具测试。`tests/release/` 需要独立环境或现有 EXE，默认明确排除；普通测试通过不代表发行验证完成。

```powershell
# 纯逻辑：不会创建 Qt 应用，也不加载正式主窗口
.\.venv\Scripts\python.exe -m pytest -q tests/unit
# 真实算法和流程；gui 子目录另覆盖 Qt 队列、按键、拖动和迟到结果
.\.venv\Scripts\python.exe -m pytest -q tests/integration
# 正式素材及指纹、离线标定/导入/兼容接口与构建清单
.\.venv\Scripts\python.exe -m pytest -q tests/resources tests/tools
```

也可用 `-m gui`、`-m resources` 等标记选组。公共夹具不从其他 `test_*.py` 导入；Qt 夹具只在 `tests/integration/gui/conftest.py` 创建，使用当前模板默认设置，并禁用真实键鼠监听。设置、缓存、日志与诊断输出使用临时目录，子进程检查也重定向运行目录，不读取个人设置。旧 cnn/cpu 偏好仅用于兼容迁移和异常状态回归。

`tests/fixtures/retrieval/cases.json` 记录输入来源、SHA256、裁剪、参考尺寸及标签依据。两组有标签输入由正式二楼底图确定性裁剪生成，另覆盖去除图标、相同结构候选并列及稀疏游戏截图；调用真实检索器，不模拟识别分数。有标签合成回放不等于游戏准确率，`game2.png` 没有独立核验的识别标签或对齐真值，只验证弱证据提示、人工确认和离线流程。定尺度测试同时保留真实匹配器的已知变换验证，检查保存尺度、原图坐标、缓存失效与版本切换。

`tests/integration/test_runtime_isolation.py` 在阻断 torch、旧引擎、tools、tests 和 Tkinter 导入的子进程中执行 `--check`、真实检索、离线投影和离屏 GUI 启停。GUI 测试使用模拟输入边界，不是游戏内检查。

发行验证使用单独的运行环境，不卸载或改造已有 `.venv`。下面的 `<运行环境 Python>` 和 `<完整发行目录中的 EXE>` 需替换为实际路径；明确执行发行组时，缺少参数或依赖会失败，不会静默跳过。

```powershell
# 在独立环境仅安装 requirements.txt；pytest 仍由开发环境运行
.\.venv\Scripts\python.exe -m pytest -q tests/release --run-release --runtime-python "<运行环境 Python>" --release-exe "<完整发行目录中的 EXE>"
```

独立环境验证要求没有 torch、pytest、PyInstaller，并执行上述四项源码入口检查；EXE 验证先创建排除个人 runtime 的临时发行目录（只读资源优先硬链接），检查旧依赖未打包，再从发行目录以外的工作目录执行资源检查及固定截图投影。复用旧 EXE 只验证该产物，不表示重新构建了最新源码。若使用本机已安装包离线组装临时环境，应记录包版本与组装方式；它验证依赖隔离，不替代从包源全新安装的验证。

## 目录与历史资料

逐项职责见 [项目目录与文件说明](项目目录与文件说明.md)。旧归档目录及本机 `workspace/releases/` 已删除；再次构建时会在新的发行目录生成产物。`.venv/`、`runtime/` 和 `workspace/` 仍按 `.gitignore` 排除提交，本机环境和运行产物不随源码提交上传。曾经提交的旧文件可通过 Git 历史查阅。

## 定版维护边界

三个失效启动器、未接入的灰度识别算法和旧楼层模板适配器已移除；当前唯一实时配准入口仍是 `template_matching/direct.py`。

运行模板共享的并发锁和资源指纹校验位于 `template_matching/resources.py`。`provenance.json` 记录该目录当前全部 Python 文件的内容指纹，构建前检查文件集合和哈希；它不是地图几何标定的替代。修改模块源码后应完成测试并同步这份代码清单。

整图结果只持有原图坐标矩阵；诊断 JSON 为兼容已有读取程序，仍输出值为 null 的 `clean_to_screen`、`screenshot_to_clean`、`observation_bbox`、`match`、`floor_detection`，不会把原图矩阵填入旧底图字段。布局耗时使用 `layout_retrieval`，同时保留相同值的历史 `classification` 键。

`configs/template_matching.json` 中的 `entries` 仍是派生尺度和参考指纹依据；旧设置读取兼容继续保留。历史诊断不能证明当前算法准确率；发行验收需要针对当前源码单独构建和验证，源码测试不能代替 EXE 验收。

主窗口与截图流程共享 `route_projection_core.game_window_is_foreground`，主窗口使用显式导入；无调用的等待／错误提示接口、旧批量分层版本选择和楼层耗时占位已退出。
