# 离线素材维护

这里保存制作原图、路线标定、结构蒙版、校正前分层图、标定截图和完整尺度推算记录。正式程序不读取本目录；运行目录可以完全不包含它。工作区和 E 盘源码目录保留这些资料，便于以后重新标定。

所有命令从项目根目录运行，记录中的 `assets/...`、`maintenance/...` 均以项目根目录为基准。识别证据中的 `sources/...` 以对应 `maintenance/maps/<地图>/<难度>/` 为基准；其 `clean` 依赖使用项目相对路径。

| 位置 | 内容 |
| --- | --- |
| `configs/template_matching.json` | 完整逐层尺度、原图尺度、几何关系和依赖指纹；用于离线审核与导出 |
| `maps/<地图>/<难度>/sources/` | 制作原图、`screen_scale_calibration.json`、`recognition_zoom50/` 配对截图及其详细标定 |
| `maps/<地图>/<难度>/recognition_scales.json` | 每张原底图的固定识别尺度与证据依赖 |
| `maps/<地图>/<难度>/routes/` | `calibration.json`、结构蒙版、凉哈皮校正前图片和 `rectification.json` |
| `maps/<地图>/<难度>/素材清单.md`、`接入核对.json` | 素材缺口、编号核对及历史接入记录（有相应记录的难度） |

正式底图、入口图片、识别小图、配置引用的最终单层路线 PNG 仍在 `assets/maps/`。`configs/scale_config.json` 每张实际图片只保存一个配准 scale；X、Y 仍由截图匹配，识别保留两个固定尺度。不要把维护配置手工复制到运行目录顶层 `configs/`。

维护工具默认暂存到 `workspace/asset_build/`，其中 `assets/` 和 `maintenance/` 两棵目录分别对应最终位置。发布图像和标定后，必须完整审核再导出运行清单：

```powershell
.\.venv\Scripts\python.exe tools/assets/imports/export_runtime_resources.py --publish
.\.venv\Scripts\python.exe tools/assets/imports/export_runtime_resources.py --check
.\.venv\Scripts\python.exe -m src --check
```

修改图片或尺度需要重新核对几何、重新生成相关识别小图，不能只改哈希。校正图导入器与导出器继续重算图片像素。历史 `tool_sha256` 保留当时的生成工具证据，不因文件搬迁改写。运行期间使用已验证快照，修改后重启生效。

历史预览和实机测试记录仍在原 `workspace/`、`runtime/`，不会因整理目录而删除。缺乏人工标签的截图只能用于结果一致性回放，不能证明识别准确率。详细流程见项目根目录 `新增布局教程.md`。
