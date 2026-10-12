# 项目协作指南

本文件适用于整个仓库。用户当前明确要求优先；默认中文沟通，标识符沿用英文。

## 当前正式范围

- 正式支持 CPU 最小缩放纯结构识别、图标与结构识别＋定尺度模板匹配。CNN、CPU/GPU 多尺度结构配准、采集训练和旧评测已移除，用户已删除 `retired_features/` 归档目录；不得从正式程序重新导入旧功能。曾经提交的旧文件和规则可从 Git 历史查阅，不能假定未提交的本机资料仍有备份。
- 正式入口 `python -m src`，由 Qt 应用调用 `src/projection_workflow.py`；识别由 `src/layout_retrieval.py` 创建，纯结构实现位于 `src/structure_layout/`，整图配准用 `src/registration/template_matching/direct.py`，共用资源校验位于 `resources.py`；旧楼层适配器已移除。
- 纯结构识别默认 CPU 最多8线程，只搜索两个最小档尺度，读取所有三层底图，不精修、不接入GPU或自动尺度；图标识别仍读取二楼底图。两者固定 0°，必须人工确认布局；匹配分数不是正确概率。不训练或加载模型，不依赖 torch/CUDA。
- 模板读取保存尺度候选，坐标为 `original_source_image`、`floor=all`，矩阵为 `original_to_screen`／`screenshot_to_original`。不识别楼层、mylocation，不执行连续尺度搜索或旧结构验收；最高分候选直接显示且保留核对提示。
- 五个模板版本为 `route_zhanshi_fast`、`route_lhp_solo`、`route_clean`、`route_lhp_solo_clean`、`route_zhanshi_fast_clean`，偏好在 `template_route_versions`。原版底图三层按原尺寸在内存拼接。不同源图必须重新截图匹配，不复用矩阵。
- 当前布局从资源目录动态读取，不硬编码数量。三层 ID 保持 `2nd`、`1st`、`b1`，布局 `layout_XX`；`route_01` 只作历史别名。资源完整不等于实机对齐。

## 修改边界

- 开工先检查相关文件和 `git status --short`，保留未提交、未跟踪和用户删除状态。不重置工作区、不覆盖无关改动。
- 优先限定 src、tools、tests、configs 搜索；不默认遍历训练数据、归档、报告和虚拟环境。
- 正式运行不得依赖 tools、tests、workspace、retired_features 或 Tkinter。通用逻辑放 src，离线素材维护放 tools；使用已有 .venv，不因普通修改重建或卸载用户环境。
- 不读取游戏内存，不自动寻路或控制角色。保留游戏前台检查、Esc 只关闭、覆盖层鼠标穿透和不抢焦点。
- 保留现有鼠标拖动跟随（试用）、Raw Input 按键队列、松键后截图延迟和日志批量刷新。拖动只按鼠标位移平移，滚轮隐藏并要求重配，不推测游戏缩放。

## 资源和归档

- 正式素材位于 `assets/maps/<map>/<difficulty>/`：clean、entrance、sources、routes。标定 JSON 和底图仍是模板尺度指纹依赖；分层 PNG 为离线维护资源，不随意删除。
- 正式资源路径保存为项目相对路径。修改底图、原图、标定或尺度必须核对对应关系及缓存失效，不能只更新哈希绕过几何检查。
- 资产工具按 calibration、imports、previews 分类，复用 `tools/common/route_calibration.py`、`calibration_assets.py` 和 `layered_assets.py`；默认写 workspace/asset_build。只有明确更新正式资产任务才发布。
- 定尺度 matcher 中 find_best_scale 只供离线尺度核对，不能接入实时流程。不要将离线素材标定的 SIFT／RANSAC 等与实时配准混淆。
- 当前不再保留 `retired_features/`、迁移清单或清理快照；不要引用其中的说明或验证脚本。研究／恢复曾经提交的旧功能须在独立副本中通过 Git 历史处理，不能让当前程序依赖旧功能。
- 不随手删除 runtime、workspace、模型备份、样本或发行产物。尊重用户已删除旧归档和 `workspace/releases/` 的状态，不为提交恢复这些目录；现有 .gitignore 规则继续保留。

## 界面和异步状态

- 耗时检索、配准和预览在后台，Qt 控件仅主线程更新，保留缓存并发保护。
- 图层先经 Qt 信号提交，随后后台合成预览；两者均校验会话、代次、布局／版本、地图开关和窗口状态。迟到结果不得恢复旧图层或确认框。
- 参考图与对齐投影为独立状态，尊重本次地图内手动显隐选择。切版本重新截图匹配，不能套用旧分层姿态复用逻辑。
- `configs/app_defaults.json` 为默认设置，`runtime/settings.json` 为个人设置。无效识别偏好归一到 pure_structure，合法 icon_structure 选择保留；投影固定 template，保留截图、热键、不透明度、模板版本等偏好。读取配置不主动落盘。

## 验证与交付

- 文档改动核对路径和命令；算法／缓存改动运行受影响模板和检索测试；Qt 改动覆盖快捷键、版本、参考图和异步状态，并检查渲染。
- 跨模块或正式入口改动运行完整 `.venv/Scripts/python.exe -m pytest -q`、`python -m src --check`、固定截图离线投影。临时目录权限受限时用新的 workspace/tmp 子目录作为 --basetemp，不覆盖旧测试产物。
- `--check` 仅检查资源／指纹，需查看 errors，不表示游戏内对齐。离线未接受退出码为 2，不可强改为成功。合成测试和模拟窗口不能称作游戏内验证。
- 测试聚焦可观察行为；不要为绕过失败改断言或删样例。已退出正式功能的专属测试已移除，混合测试保留当前功能的回归覆盖。
- 测试分为 tests/unit、integration（含 gui）、resources、tools、release；共用辅助对象放 support，禁止从 test_*.py 导入夹具。纯逻辑测试不创建 Qt 应用；GUI 夹具集中到 integration/gui/conftest.py，禁用真实键鼠监听并隔离个人设置、缓存与输出。
- 默认 pytest 排除 release；发行验收须显式 --run-release 并提供 --runtime-python、--release-exe。缺失条件报失败，不静默跳过。记录是否全新安装、离线复制运行依赖或复用已有 EXE；三者不能混称新构建验收。
- 图像回放来源/指纹/标签依据记录在 tests/fixtures/retrieval/cases.json。底图裁剪合成样例可验证已知布局，缺少人工标签的游戏截图不能断言识别准确率；不恢复多尺度拒绝门槛。
- 使用说明维护 README，逐文件职责维护 项目目录与文件说明.md，新增素材流程维护 新增布局教程.md。文档中的目录、命令和链接必须与当前项目一致，不承诺已删除归档仍可用。
- 模板模块 `provenance.json` 覆盖当前目录全部 Python 源码；修改该模块后同步代码指纹并检查构建拒绝陈旧指纹。它不能代替素材几何校验。
- 版本统一读取 src/version.py（当前 1.1）。构建入口 tools/release/build_windows_exe.py，输出 workspace/releases/windows_<时间>/IdentityVRouteAssistant；完整文件夹分发。排除归档、模型、训练包、个人状态和旧分层 PNG；保留实际原图与标定依赖。旧产物不自动改写。
- 交付写清修改、迁移位置、实际验证和未验证限制；不得把历史测试数量、模型准确率或旧性能写成保证。
