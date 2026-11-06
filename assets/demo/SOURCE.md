# demo_input.mp4 — 来源与许可

| 项目 | 内容 |
|---|---|
| 文件 | `demo_input.mp4` (20 s, 960x540, 29.97 fps, mp4v, 无音轨) |
| 原始视频 | *Science Speaks - Talking to Women in Science (Liz, Epidemiologist)* |
| Commons 页面 | https://commons.wikimedia.org/wiki/File:Science_Speaks_-_Talking_to_Women_in_Science_(Liz,_Epidemiologist).webm |
| 原始文件 | https://upload.wikimedia.org/wikipedia/commons/7/7b/Science_Speaks_-_Talking_to_Women_in_Science_%28Liz%2C_Epidemiologist%29.webm (1280x720 WebM, 283 s) |
| 作者 | National Institute for Occupational Safety and Health (NIOSH, 美国疾控中心 CDC 下属机构) |
| 出镜者 | Elizabeth A. Masterson, PhD, NIOSH 流行病学家 |
| 日期 | 2013-11-04 |
| 许可 | **Public domain** (美国联邦政府雇员职务作品, 17 U.S.C. §105; Commons 标注 "Public domain", AttributionRequired = false) |

## 处理方式
用 OpenCV 从原片截取 110.5 s – 130.5 s 这一段连续镜头 (中间没有剪辑/字幕卡),
裁剪区域 x = 313..1273, y = 0..540 (原生分辨率, 不缩放), 重新编码为 mp4v。
脚本: `_scratch/fx_make_demo_input.py`。

虽然公有领域作品不要求署名, 这里仍注明出处, 也不暗示 NIOSH/CDC 或出镜者认可本项目。
