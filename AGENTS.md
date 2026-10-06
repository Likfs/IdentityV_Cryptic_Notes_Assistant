# 项目协作指南

本文件适用于整个仓库。用户在当前任务中的明确要求优先于本文件；子目录有更具体的指令时，在对应范围内遵循。默认使用中文沟通，代码标识符保持现有英文风格。

## 项目定位

这是《第五人格》的 Windows 地图布局识别与路线投影工具。CPU/GPU 主流程为：游戏截图 → 布局识别并锁定 → 楼层确认 → 单帧多尺度结构配准→ 按标定关系透明覆盖显示；定尺度模板匹配直接匹配并投影整张三层原图。

- 当前正式入口为 `src/registration/multiscale_structure/runtime.py`，由 `src/projection_workflow.py` 调用；不要误把旧 `runtime_registration.register_screenshot()` 当作当前默认。
- 应用默认使用定尺度模板匹配（`registration_backend=template`）。多尺度结构配准选择 `auto` 时 GPU 优先，实现位于 `src/registration/multiscale_structure/gpu/`；原 CPU 入口完整保留为 `src/registration/multiscale_structure/cpu/runtime.py`，其 `multiscale_structure/cpu/search_core/`、`multiscale_structure/cpu/engine.py` 和 `registration_core/` 依赖也须保留。设置 `registration_backend=cpu` 强制原 CPU，`auto` 优先 GPU 并在不可用或异常时回退；正常拒绝不能触发换后端重试。GPU 代码延迟导入，CPU 不依赖 CUDA，模板动态缓存分离。
- GPU 使用说明见 `README.md` 的“配准与投影”。用户于 2026-10-01 明确要求删除整个 archives，历史备份已清除，当前不可用于本机回退。
-多尺度结构配准保持已复核的独立单帧行为，不读写可信历史；路线图片墙线、覆盖率检查及路线补充确认按用户要求停用。底图／截图的几何和像素证据校验仍保留。
-多尺度结构配准的 `joint_acceptance.py` 仅为类别一致率轻微不足提供受约束补充接受；不能按通过检查的数量抵消其他失败。CPU/GPU 开图延迟默认 150 ms、可设置，只以可靠楼层识别为准，不再识别 mylocation；楼层复用仍须同会话、同窗口且新鲜，截图后再次确认。详见 `README.md` 的“配准与投影”。

- `registration/multiscale_structure/cpu/search_core/` 是多尺度结构配准的正式搜索和公共算法依赖，不能删除。CPU/GPU 结构见 `项目目录与文件说明.md`，使用说明见 `README.md`。目录、标识和说明按算法职责命名，不恢复历史版本目录。

- 布局识别可选 PyTorch 自定义 CNN 或 `src/layout_retrieval.py` 的固定 0° 图标与结构匹配，默认图标与结构匹配，已有合法 CNN 偏好保留。后者读取当前地图／难度的二楼底图，必须人工确认正式布局后再投影，匹配分数不能当作正确概率。GPU 配准结合 CUDA 颜色／稀疏评分与 CPU OpenCV 结构搜索和几何校验，原 CPU 后端仍可独立运行。
- 路线来自预先制作的图片。自动寻路、角色控制、游戏内存读取不属于现有功能，不要在无关任务中引入。
- 当前已标定分层资源为厄运之女／噩梦的 27 个布局、每布局三层，覆盖原版底图、凉哈皮、展十速刷三种显示版本。模板保存尺度已覆盖 layout_01～layout_27，共 81 条楼层记录；图标与结构识别动态覆盖这些布局，正式 CNN 模型仍覆盖 layout_01～layout_21，不能据资源齐全宣称 CNN 新类别已训练。配置校验不代替游戏内对齐验证。扩展时从资源目录读取布局，不把数量硬编码进通用逻辑。
- 以当前代码、配置和实际资源为准。历史说明已合并到两份根目录文档并清理；不能仅凭历史方案宣称某功能已启用或某指标已达成。

- 正式版已接入 `template`（定尺度模板匹配），与 CPU/GPU 并列。实现 `registration/template_matching/direct.py` 使用保存尺度匹配三层整张原图，跳过楼层与 mylocation，坐标为 original_source_image、floor=all，不能冒充 clean_to_screen。最高分候选直接显示，保留候选提示，不套用多尺度结构配准验收。
- 模板模式提供展十速刷路线图、凉哈皮单人速刷路线图及原版干净底图；底图三层原尺寸在内存拼接，默认展十，偏好独立保存在 `template_route_versions`。凉哈皮每层固定尺度由现有标定离线推算；仅比较保存候选、不连续搜尺度，候选楼层不是识别结果。切换整图版本须重新截图匹配，不能复用另一原图矩阵；尺度推算及资源检查不等于游戏内对齐验证。
- 模板模式另提供 `route_lhp_solo_clean`（凉哈皮单人速刷路线图底图版）及 `route_zhanshi_fast_clean`（展十速刷路线图底图版），读取同名 sources 目录。底图版逐层验证原图坐标一致后继承固定尺度与搜索区域，保存独立源图指纹及参考版本依赖；原图、标定或尺度改变后须重新导入，不作为 CPU/GPU 分层版本。
- 定尺度模板匹配实时投影图层先经 Qt 信号提交，随后后台合成预览；覆盖层提交耗时不是物理显示延迟。迟到图层和预览均检查会话、代次及状态。截图延迟可配置，默认 150 ms。
- 路线目录为 `routes/<variant_id>/<layout_id>/`，凉哈皮统一 `route_lhp_solo`，旧 `route_01` 仅作为设置/命令兼容别名。游戏识别图在 `assets/game_ui/`，程序背景和图标在 `assets/app_ui/`。
- 正式配准接入及资源分类说明见 `README.md` 和 `项目目录与文件说明.md`。

## 开始工作与改动范围

- 先检查相关文件及 `git status --short`，保留用户已有的未提交、未跟踪和删除状态；不要覆盖与任务无关的改动。
- 根据任务读取相关模块，不必每次遍历所有报告、样本、备份和虚拟环境。搜索代码优先限定 `src/`、`tools/`、`tests/`、`configs/`。
- 保持改动围绕当前问题；不顺带重写界面、迁移目录、重训模型或重新标定全部资产。
- 完成已授权的修改和验证；常规实现选择无需逐项询问。遇到影响需求方向的信息缺口再澄清。

## 代码与文件职责

| 路径 | 职责与修改入口 |
| --- | --- |
| `src/route_projection_app.py` | 正式 PyQt6 应用、任务调度、会话状态、快捷键及结果应用 |
| `src/route_projection_qt_ui.py` | Qt 控件、确认框、设置页、提示窗、投影窗和参考图窗 |
| `src/route_projection_core.py` | 游戏窗口、截图坐标换算、地图按键状态和指标显示辅助 |
| `src/projection_workflow.py` | 分类、配准、投影、独立路线校验和诊断的流程组合 |
| `src/app_catalog.py` | 地图／难度／路线版本目录、设置读写和正式布局判定 |
| `src/layout_classifier.py` | CNN、数据集、预处理、分类推理 |
| `src/layout_retrieval.py` | 二楼底图图标与结构检索、正式布局候选、内容指纹和索引缓存 |
| `src/floor_indicator.py`、`src/map_presence.py` | 楼层模板匹配、地图界面标识检测 |
| `src/capture/` | 截图区域、运行时截图及热键、固定区域样本采集器 |
| `src/registration/` | 配准策略、失败恢复、多帧证据管理 |
| `src/registration_core/` | 分割、模板预处理、索引、匹配、几何复核等算法 |
| `src/projection/` | 路线资产加载、坐标变换、合成和原生覆盖层 |
| `tools/demos/` | Tkinter 调试界面；不能替代正式程序入口 |
| `tools/` | 开发工具、采集器与 Demo；其余按 `training/`、`assets/`、`evaluation/` 分类 |
| `tests/` | pytest 测试，包括算法、资源逻辑和 Qt 状态回归 |
| `configs/` | 地图目录、采集设置、用户设置等 JSON |
| `assets/` | 正式地图、入口图、路线资产、识别模板和界面图片 |
| `workspace/` | 分类样本、JSONL 索引及可重建缓存 |
| `models/releases/` | 正式运行模型 |
| `workspace/checkpoints/` | 训练输出、验证检查点和历史模型 |
| `workspace/reports/`、`runtime/logs/` | 诊断、评测、预览和日志输出 |
| `tests/fixtures/registration/` | 固定配准复现输入 |
| `workspace/tmp/` | 临时维护脚本与暂存内容；archives 已按用户要求删除 |

把可复用算法放在相应 `src/` 模块，界面层负责交互和展示，脚本负责执行入口。批量标定复用 `tools/common/route_calibration.py`；工具目录已移除 comparison 和 experiments，evaluation 仅保留工具说明中列出的六个核心评测脚本。

资产工具按 `tools/assets/calibration/`、`previews/`、`imports/`、`cache/` 分类，命令使用新路径。采集器和训练维护共用 `tools/common/sample_store.py`，不从采集器界面导入样本服务。采集默认参数为 `tools/configs/capture_defaults.json`，个人参数在 `workspace/tool_state/capture.json`，通过 `tools/common/capture_config.py` 兼容读取旧路径。路线 Demo 的投影共用 `tools/demos/projection_service.py`，调用正式多尺度结构配准workflow；其布局门槛与手动楼层仍属 Demo 交互。

## 环境与常用命令

目标环境为 Windows 10/11、Python 3.10+、PowerShell。从项目根目录执行命令，优先使用已有 `.venv`；不要为普通修改重建用户环境。

```powershell
# 仅在环境缺失或依赖需要安装时执行
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt

# 资源清单检查、启动正式程序
.\.venv\Scripts\python.exe -m src --check
.\.venv\Scripts\python.exe -m src

# 自动化测试
.\.venv\Scripts\python.exe -m pytest -q

# 使用固定样例验证投影，输出诊断文件
.\.venv\Scripts\python.exe -m src --project-once tests/fixtures/registration/game2.png --layout layout_05 --floor 2nd --route-version route_lhp_solo --output reports/route_projection/manual_check

# 路线资产质量审计
.\.venv\Scripts\python.exe tools/evaluation/audit_route_assets.py
```

完整应用需要 PyTorch，当前 `requirements.txt` 已包含该依赖；`requirements-training.txt` 复用正式依赖，`requirements-dev.txt` 额外添加 pytest。配准应保留 CPU 可用性，不为速度优化引入强制 GPU 依赖。

`--check` 仅检查模型存在与资源清单状态；退出码为 0 不代表图像质量已验证。检查输出中的缺失模型、`REVIEW` 和各版本楼层数量。离线投影未通过时退出码为 2，需检查拒绝原因，不能直接改成成功。

## 资源、数据与模型约定

- 正式资源目录：`assets/maps/<map_id>/<difficulty_id>/`。`clean/<layout_id>/` 保存三层底图；`entrance/` 保存布局确认图；`sources/<variant_id>/` 保存原图；`routes/<variant_id>/<layout_id>/` 保存运行图层与 `calibration.json`。
- 楼层 ID 保持 `2nd`、`1st`、`b1`；布局保持 `layout_XX`。路线 ID 保持 `route_clean`、`route_lhp_solo`、`route_zhanshi_fast`，修改显示名称不应破坏这些持久化 ID。
- `route_clean` 直接引用底图并使用单位矩阵。展十的 `*_structure.png` 属于运行时校验依赖，不能当作临时预览删除。
- 保存到配置和标定文件的资源路径使用项目相对路径；运行时通过项目根目录解析，避免写入本机盘符。
- 新布局需要分类样本、模型类别、三层底图和对应路线标定。仅增加图片目录不等于完整支持。
- 采集器按现有样本目录分配新编号，不从地图资产推断编号。新克隆缺少旧样本时，不要假定下一个编号必然是 `layout_22`。
- JSONL 从现存已归档图片重建；不要让索引继续引用已删除图片。维护入口在 `tools/training/`。采集器删除样本时直接删除 PNG 并更新 JSONL，不保留删除副本；独立入口为 `delete_sample.py`。
- 训练输出留在 `workspace/checkpoints/`。发布模型属于明确的模型更新任务；核对标签、预处理、独立验证及回滚文件后才更新 `models/releases/` 或目录配置。
- 替换底图、路线或特征提取逻辑时，更新相应标定／缓存指纹或版本，验证缓存失效。不能跨布局、楼层或不同源图片错误复用缓存。
- 遵循 `.gitignore`：本机设置、训练样本、缓存、报告、日志和备份默认不提交。需要新增固定测试样例时，使用少量可复现输入，不把整批诊断搬进仓库。
- 不为节省磁盘随手删除 `workspace/`、`models/`、`reports/`、`archives/` 中的现有文件；资源整理应是明确任务的一部分。

## 配准与投影修改约束

- 明确坐标空间：物理屏幕、游戏客户区、截图 ROI、观测裁剪区、预处理底图、原始底图、路线资产。矩阵命名标明方向，正确处理裁剪偏移与预处理缩放。
- 运行时主配准使用等比例缩放和平移；离线素材标定可使用 SIFT、RANSAC、仿射／单应矩阵及 ECC。不要混淆两个流程的假设和误差单位。
- 路线投影关系为 `asset_to_screen = clean_to_screen @ asset_to_clean`，矩阵相乘前统一为齐次形式。保留 BGR/BGRA 与 RGB/RGBA 的转换边界。
- 未知迷雾、已知背景、可见结构和忽略区域必须保持区别。未知区域不能直接充当“没有房间／走廊”的负证据；不能按候选底图反向制造截图证据。
- 配准接受需结合结构、语义、边缘、空间竞争和缩放稳定性。多尺度结构配准使用已复核的容差、竞争分支重验、可见区域稳定性及扩展尺度约束；不要重新启用已移除的路线图片复核。
- 区分可信历史、未确认候选和软位置提示。保留对弱证据、路线补充确认、缩放放宽及局部证据写入可信记忆的限制，防止一次误配准污染后续帧。
-多尺度结构配准的三种显示版本均复用底图配准结果；路线资产只提供显示内容，不作为独立验收证据。之前版本的路线复核代码只用于历史回归。
- 修改实验开关前检查 `runtime_registration_config()` 的实际默认值。开启原本关闭的候选检索或搜索策略需要相应回归证据；“代码存在”不等于“默认启用”。
- 当前正式流程经多尺度结构配准配准接受后显示完整路线；`clip_local_layer()` 用于离线诊断。不要把局部证据范围和最终显示范围混为一谈。
- 候选之间的路线 P95 位移是歧义／敏感度指标，不是人工真值误差，也不是严格置信区间。

## 界面与异步状态约束

- 正式应用和固定区域采集器使用 PyQt6，Demo 使用 Tkinter。正式运行时通用服务保持不依赖 Tkinter，避免从 Demo 导入服务代码。
- 耗时识别、配准和预热在后台完成；Qt 控件在主线程更新。保留模型、模板和路线缓存的并发保护。
- 地图关闭、楼层变化、布局切换或会话重置后，迟到结果不得恢复旧图层、旧确认框或旧证据。
- 保留游戏前台窗口检查、地图界面检测、Esc 只关闭的语义，以及覆盖层鼠标穿透和不抢焦点行为。
- 参考路线图无需配准，但需要有效布局和楼层；它与已对齐投影是不同窗口与状态。尊重用户本次地图内的手动显隐选择。
- 三版本即时切换可复用有效姿态，但仍需校验新资产，并检查布局、楼层、会话与地图状态是否匹配。
- 正式布局候选无论置信度与分差多少均允许人工确认；`app_catalog.prediction_decision()` 的门槛仅用于判定置信度和允许跳过确认框自动锁定，低置信度即使开启跳过确认也须弹框。Demo 的探索比例门槛不要意外带入正式流程。
- 新增设置保留旧配置的默认值和兼容加载，界面文案使用中文。行为调整同步更新相关使用说明。

## 按改动选择验证

| 改动 | 验证要求 |
| --- | --- |
| 文档、注释、低影响文案 | 检查路径、命令、拼写和描述与实际代码一致；无需重跑算法基准 |
| 采集、分类、训练 | 运行对应的 `test_fixed_roi_collector.py`、`test_layout_classifier.py`、`test_training_modes.py` 等受影响测试 |
| Qt 界面、快捷键、会话 | 运行应用、Qt、参考图和版本切换相关测试；布局改动检查实际渲染 |
| 配准、分割、恢复、证据 | 运行对应算法回归测试；改变接受策略或搜索范围时增加固定回放和已知变换验证 |
| 资产、目录配置、路径 | 运行 `--check`，检查质量审计输出与受影响楼层的标定／投影预览 |
| 跨模块行为或正式运行链路 | 运行完整 `python -m pytest -q`，并选择固定样例做离线投影检查 |

示例定向测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_route_projection_app.py tests/test_route_projection_qt_ui.py tests/test_reference_map_ui.py tests/test_live_version_switch.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_first_sparse_registration.py tests/test_sparse_pose_review.py tests/test_credible_geometry.py tests/test_registration_recovery.py tests/test_temporal_evidence.py
```

配准评测入口包括 `tools/evaluation/benchmark_first_registration.py`、`benchmark_first_synthetic.py` 和 `verify_gpu_production.py`。执行前查看 `--help`，确认输入清单、后端和输出路径；两个 benchmark 默认 `--backend cpu`，`auto` 为 GPU 优先并记录回退，历史搜索开关已停用。部分脚本依赖本机运行诊断中的历史截图，新克隆可能没有这些数据。缺失时报告限制，不将空回放或旧报告视为本次验证通过。

比较性能时固定输入、线程数、配置与历史状态，区分冷启动、模板准备、首次搜索和历史复用。统计接受、拒绝、位置差异及 P50/P95；只有通过结果的耗时不能代表全部输入。旧算法结果不等于真值，已知变换合成测试也不等于实机准确率。

测试聚焦可观察行为和真实回归，不仅复写实现。不要为绕过失败而删断言、改期望结果或移除样例。测试通过后，除非有新修改或未解决问题，不重复运行同一批验证。

## 文档与交付

- 使用、安装、训练、资产及评测看 `README.md`；目录与逐文件职责看 `项目目录与文件说明.md`；新增布局操作看用户明确要求编写的 `新增布局教程.md`，覆盖三种配准与三种路线版本。按任务需要读取。
- 修改用户可见行为、配置、目录或算法默认策略时，同步维护这两份文档；AGENTS.md 保留协作规则。实验细节与失败案例放 `workspace/reports/`，长期结论和复现方式合并到 README，不新增独立源码说明文档。
- 用户于 2026-10-01 将项目定为 1.0，2026-10-06 更新为 1.1；统一版本来源为 `src/version.py`，UI、CLI、Qt 应用信息、Windows 文件／产品版本与构建记录同步读取。旧发行产物不自动改写，第三方文档／许可证和生成产物说明保留。
- 最终说明写清改了什么、为何修改、实际运行的验证及结果、未验证的限制。注明是否做过实机检查；不把模拟窗口测试称作游戏内验证。
- 不将历史测试数量、固定性能数字或模型验证集准确率写成永久保证。状态变化后维护本文件，保持项目指令简洁且可执行。

## 正式运行与开发工具边界

- 正式启动使用 `python -m src`，根目录批处理保持可用。正式运行不得依赖 tools、workspace、tests 或 Tkinter。
- `configs/app_defaults.json` 为默认设置，`runtime/settings.json` 为个人设置；不再从采集器或 Demo 导入默认值。缓存、日志、诊断写入 runtime。
- 采集、训练、评测、Demo 和算法对比入口均位于 tools；数据、检查点、工具设置、缓存和报告位于 workspace。
- 用户于 2026-10-01 删除 packaging 发行构建工具与 assets/maps/inventory.json 汇总清单，当前正式程序不依赖它们；不要将历史构建命令作为可用入口。保留的资产整理工具仍可生成清单，默认写入 workspace/asset_build。
- 用户随后于 2026-10-01 要求构建 EXE，现构建入口为 `tools/release/build_windows_exe.py`，独立构建依赖在同目录 `requirements-build.txt`。输出 `workspace/releases/windows_<时间>/IdentityVRouteAssistant/`，为完整功能文件夹发行版，须整体分发；资源按白名单收集，不打包个人 runtime 或开发数据，不恢复旧 packaging 工具。
- 素材生成默认写入 workspace/asset_build；提供 --publish 的工具仅在用户明确要求更新正式资产时使用此参数。尺度导入默认生成暂存配置。
- 迁移清单和验证原位于 workspace/maintenance；该目录及迁移前 archives 快照已于 2026-10-01 按用户要求清除。
- 用户于 2026-10-01 清除 workspace 的 tmp、cache、reports、releases、checkpoints、asset_build、maintenance（含临时副本中的嵌套归档）；训练样本、索引和 tool_state 保留。历史文件中的这些路径不代表现存内容，开发工具仍可按需生成新产物。
- 用户于 2026-10-02 清除独立识别实验副本、指定测试临时产物及两套旧 EXE。正式图标与结构识别、两个 Demo、自动化测试源码与固定样例、构建工具、标定报告及运行诊断保留；workspace/tmp/repository_v1_upload 中的 Git bundle、索引及提交记录保留。
