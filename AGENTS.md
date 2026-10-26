# 项目协作指南

本文件适用于整个仓库。用户当前明确要求优先；默认中文沟通，标识符沿用英文。

## 当前正式范围

- 正式支持 CPU 固定缩放纯结构识别、图标与结构识别＋定尺度模板匹配。CNN、CPU/GPU 多尺度结构配准、采集训练和旧评测已移除，用户已删除 `retired_features/` 归档目录；不得从正式程序重新导入旧功能。曾经提交的旧文件和规则可从 Git 历史查阅，不能假定未提交的本机资料仍有备份。
- 正式入口 `python -m src`，由 Qt 应用调用 `src/projection_workflow.py`；识别由 `src/layout_retrieval.py` 创建，纯结构实现位于 `src/structure_layout/`，单层配准用 `src/registration/template_matching/direct.py`，共用资源校验位于 `resources.py`；旧楼层适配器已移除。
- 纯结构识别默认 CPU 最多8线程，每张底图只搜索一至两个已配置的固定尺度（未单独标定时沿用原两个最小档值）；困难识别支持最小与 50% 两档，模板配准仍仅使用最小尺度。读取当前难度各布局的全部有效楼层底图，不精修、不接入GPU或自动尺度；图标识别仍读取二楼底图。两者固定 0°，默认人工确认布局，可显式开启 `auto_select_layout` 自动锁定最高分有效候选；匹配分数不是正确概率。不训练或加载模型，不依赖 torch/CUDA。
- 模板读取保存尺度候选，坐标为 `single_floor_source_image`、`floor` 为胜出候选 `2nd`／`1st`／`b1`，矩阵为 `original_to_screen`／`screenshot_to_original`。不独立识别楼层按钮或 mylocation，自动比较已保存有效尺度的单层图，不执行连续尺度搜索或旧结构验收；最高分候选直接显示且保留核对提示。
- 噩梦五个模板版本为 `route_zhanshi_fast`、`route_lhp_solo`、`route_clean`、`route_lhp_solo_clean`、`route_zhanshi_fast_clean`，偏好在 `template_route_versions`。正式配准尺度在 `scale_config.json`，图片归属在 `runtime_resources.json`，读取后适配为内部 `floor_variants`；原版底图各层独立读取，不拼接。不同源图必须重新截图匹配，不复用矩阵。
- 当前布局从资源目录动态读取，不硬编码数量。有效楼层由难度 profile 的 `floors` 声明：噩梦为 `2nd`、`1st`、`b1`，困难只有 `2nd`、`1st`，不能为困难补空白地下室。布局 `layout_XX`；`route_01` 只作历史别名。资源完整不等于实机对齐。
- 困难使用 `route_zhanshi_full`（展十全图路线）、`route_zhanshi_full_clean`（展十全图路线底图版）与 `route_clean`，不冒用噩梦速刷版本；各难度的显示版本由 `template_versions` 声明。困难 24 号暂缺两层底图，保留缺号及原图。入口图和实机截图后补；默认以候选对应楼层的底图辅助人工确认，显式开启自动选择时可省略确认框。未核对屏幕尺度的单层记录使用 `scale_status: pending`、`saved_scale: null`，资源仍核对指纹，仅匹配已有尺度的楼层，全部 pending 时投影明确拒绝，不套用噩梦尺度。

- 困难 `layout_05/1st` 的用户标注截图提供直接尺度 `3.178`；其余 57 张底图经公共结构比例及多路径一致性核对后独立推算，保存为 `derived_geometry`。离线识别配置 `recognition_scales.json` 与模板维护记录绑定 `sources/screen_scale_calibration.json` 及全部参与底图指纹，导出时完整审核；正式程序读取 `configs/runtime_resources.json` 中的固定尺度与实际图片指纹。变更素材须重新推算，不能只更新哈希。两种识别均读取对应底图固定尺度，图标模式仍只读二楼。固定截图和素材推算不代表所有楼层、其他缩放档或实时游戏操作已验收；16:50～16:51 的截图由用户确认为无可靠布局标签，不把目录中的选中布局当真值。

## 修改边界

- 困难纯结构识别使用 `recognition/index.json` 绑定的约 47.2% 等比例小图，共用最小 `1.50` 与 50% 约 `0.61762837` 两个尺度；原 `recognition_scales.json` 保留逐原图几何依据及图标识别尺度。小图墙线过滤长度按缩图比例换算，粗采样固定 30.8%（精确值 `0.30792678925763023`，用户已确认），不增加精修。原图、确认框和最小缩放模板配准不缩图。源素材／尺度变更后须重新生成完整小图，发布时从原图重算像素核对；不能只更新哈希或复用旧小图。小图坐标与原图坐标明确区分，识别姿态不能传入配准。

- 困难识别额外支持 50% 固定档，`recognition_scales.json` 的 `scale_modes` 与 `zoom_derivation` 绑定配对截图和最小尺度推算证据。配准仍只用最小尺度；F8 确认只传布局，用户缩到最小后重新截图配准。当前两组截图仅证明跨缩放候选一致性，不是所有布局两层的实机真值。

- 开工先检查相关文件和 `git status --short`，保留未提交、未跟踪和用户删除状态。不重置工作区、不覆盖无关改动。
- 优先限定 src、tools、tests、configs 搜索；不默认遍历训练数据、归档、报告和虚拟环境。
- 正式运行不得依赖 tools、tests、workspace、retired_features 或 Tkinter。通用逻辑放 src，离线素材维护放 tools；使用已有 .venv，不因普通修改重建或卸载用户环境。
- 不读取游戏内存，不自动寻路或控制角色。保留游戏前台检查、Esc 只关闭、覆盖层鼠标穿透和不抢焦点。
- 保留现有鼠标拖动跟随（试用）、Raw Input 按键队列、松键后截图延迟和日志批量刷新。拖动只按鼠标位移平移，滚轮隐藏并要求重配，不推测游戏缩放。
- 地图、识别／投影、参考图及计数快捷键共用 Raw Input 队列，不重复注册系统热键。支持小键盘数字、运算符、小数点及独立 `NumEnter`，主键盘数字／Enter 分开；数字和小数点按扫描码归一化，NumLock 不改变绑定。`NumEnter` 的内部事件码不能传给 Windows 注册或状态轮询 API。Esc 保留关闭／取消语义，不作为可录制操作键。

## 资源和归档

- 模板配准按用户确认的固定资源策略运行：配置与难度楼层声明在进程首次使用时固定，各布局/版本首次加载时校验发布配置与实际运行图片并保存压缩字节快照；后续配准及缓存淘汰后的解码只用已验证快照，修改素材/尺度/模板配置后重启生效。原图像素、保存尺度和匹配算法保持不变；不得恢复逐帧全量磁盘校验。`--check` 和离线维护接口仍检查磁盘当前内容，初次校验失败不得发布半批快照。临时快照由进程自动清理，不随发行包分发。
- 困难纯结构识别由 `create_layout_retriever` 为识别小图 profile 启用 `fixed_resources`：首次准备完整核验并构建内存索引，发布前复核文件与底图目录链接，成功后不再逐请求读取资源。素材／尺度修改后重启生效；关闭识别器释放索引，重新创建时完整核验。直接创建 `PureStructureRetriever` 默认保留逐请求核验，供离线维护；工厂也可显式 `fixed_resources=False`。`--check` 保持检查磁盘，不得把初次失败或取消的半批索引固定下来。不改变噩梦、图标识别、两档尺度或配准。

- 正式素材位于 `assets/maps/<map>/<difficulty>/`：clean、entrance、recognition 和配置引用的 routes 单层 PNG。标定 JSON、制作原图、结构蒙版、校正前图片和标定截图集中在 `maintenance/maps/<map>/<difficulty>/`；完整模板维护配置在 `maintenance/configs/`。完整依赖由导出工具审核，正式运行不依赖 maintenance。维护工具默认暂存 assets 与 maintenance 两棵目录；保留历史工具指纹和几何证据，不能随意删除。
- 正式资源路径保存为项目相对路径。修改底图、原图、标定或尺度必须核对对应关系及缓存失效，不能只更新哈希绕过几何检查。
- 资产工具按 calibration、imports、previews 分类，复用 `tools/common/route_calibration.py`、`calibration_assets.py` 和 `layered_assets.py`；默认写 workspace/asset_build。只有明确更新正式资产任务才发布。
- 定尺度 matcher 中 find_best_scale 只供离线尺度核对，不能接入实时流程。不要将离线素材标定的 SIFT／RANSAC 等与实时配准混淆。
- 当前不再保留 `retired_features/`、迁移清单或清理快照；不要引用其中的说明或验证脚本。研究／恢复曾经提交的旧功能须在独立副本中通过 Git 历史处理，不能让当前程序依赖旧功能。
- 不随手删除 runtime、workspace、模型备份、样本或发行产物。尊重用户已删除旧归档和 `workspace/releases/` 的状态，不为提交恢复这些目录；现有 .gitignore 规则继续保留。

## 界面和异步状态

- 耗时检索、配准和预览在后台，Qt 控件仅主线程更新，保留缓存并发保护。
- 图层先经 Qt 信号提交，随后后台合成预览；两者均校验会话、代次、布局／版本、地图开关和窗口状态。迟到结果不得恢复旧图层或确认框。
- 参考图与对齐投影为独立状态，尊重本次地图内手动显隐选择。切版本重新截图匹配，不能套用旧分层姿态复用逻辑。
- “下一局”清零宝箱计数并重置布局、地图开关、参考图、截图和配准预览，即使计数为零也执行；撤销仅恢复计数。取消旧任务并清除待确认、延迟截图和待投影，任务回调按身份及会话检查，迟到结果不能恢复上一局状态或清除新任务的忙碌状态。保留已验证资源缓存和个人偏好。
- `auto_select_layout` 默认关闭，仅显式布尔值 true 开启；旧 `skip_layout_confirmation` 继续忽略，不迁移为自动选择。自动选择仅锁布局，不复用识别姿态或自动截图投影；困难识别后仍须缩到最小并重新截图配准。
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
- 版本统一读取 src/version.py（当前 1.1）。构建入口 tools/release/build_windows_exe.py，输出 workspace/releases/windows_<时间>/IdentityVRouteAssistant；完整文件夹分发。排除归档、模型、训练包、个人状态和未引用的旧分层 PNG；包含 floor_variants 引用的单层图片；只打包实际运行图片与发布清单；完整制作原图和标定依赖保留离线，构建前审核。旧产物不自动改写。
- 交付写清修改、迁移位置、实际验证和未验证限制；不得把历史测试数量、模型准确率或旧性能写成保证。

## 运行资源发布契约

- `configs/scale_config.json` 是配准尺度表，`maps[图片路径].scale` 每图仅一个固定值；`configs/runtime_resources.json`（schema_version 2）保存自动生成的图片分组／无尺度元数据及独立识别两档配置。运行读取后适配为原匹配记录，数值不取整、候选顺序和像素处理不变，X、Y 仍由截图匹配。`maintenance/configs/template_matching.json`、逐难度 `recognition_scales.json` 及详细制作记录保留作离线输入。维护后使用 `tools/assets/imports/export_runtime_resources.py --publish` 成对发布，重启生效；不得手工只改运行清单指纹。
- 导出完整验证依赖和参考记录、从原图重算识别小图逐像素核对，并拒绝导出期间输入变化。正式包只保存实际图片及尺度/状态摘要；不自动回退到离线资料修补损坏的已发布包。
- `python -m src --check` 检查当前运行包；导出工具 `--check` 审核完整离线资料并比较生成结果。发布 EXE 前必须执行完整审核。维护资料保留原位，不删除或搬到临时目录。
- 运行格式变化允许配置/来源记录指纹改变；尺度、所有候选数值、坐标、双向矩阵及投影像素必须回放一致。无离线资料的独立运行目录验收不能称作新的 EXE 构建或实机验收。
