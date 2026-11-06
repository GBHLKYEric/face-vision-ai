## 总览|人脸识别与特效详解|总览 {#home|总览}

这页把 FaceVision 项目（PDF《字节跳动智能视觉AI：人脸识别与特效技术》四个阶段的完整实现）从原理到每一行关键代码讲一遍：12 个专题、几十道带数值的例题，最后是一页必背清单。代码在 {{REPO_LINK}}。

全部结果都是在一台没有独立显卡的笔记本（Intel Core Ultra 5 225H + Arc 130T 核显）上实测的，数字都能用仓库里的脚本复现。

<ul class="how"><li><b>点上面的标签跳转</b>第一行是专题，第二行是当前专题的小节，往下读时会自动跟着高亮。</li><li><b>每个专题同一顺序</b>知识要点 → 代码精读（摘自仓库真实代码）→ 例题（数字都用代码复算过）。</li><li><b>紫色方框是最终答案</b>绿色标签是口诀、类比和易错点，黄色是需要特别注意的地方。</li></ul>

<div class="key"><span class="lab">核心结果</span>

{{HOME_RESULTS}}

</div>

### 整条流水线 {#home-pipe}

<figure class="fig"><div class="scroll"><svg class="dg" style="width:100%;min-width:680px" viewBox="0 0 760 250" role="img" aria-label="人脸识别与特效流水线">
<rect class="bx" x="10" y="95" width="88" height="50" rx="8"/><text x="54" y="118" text-anchor="middle" class="nm">照片/视频</text><text x="54" y="135" text-anchor="middle" class="qt">任意大小</text>
<rect class="bx hot" x="130" y="95" width="100" height="50" rx="8"/><text x="180" y="118" text-anchor="middle" class="nm">检测 SCRFD</text><text x="180" y="135" text-anchor="middle" class="qt">框 + 5 点</text>
<rect class="bx hot" x="262" y="95" width="100" height="50" rx="8"/><text x="312" y="118" text-anchor="middle" class="nm">对齐 Umeyama</text><text x="312" y="135" text-anchor="middle" class="qt">112×112</text>
<rect class="bx hot" x="394" y="95" width="110" height="50" rx="8"/><text x="449" y="118" text-anchor="middle" class="nm">识别 ArcFace</text><text x="449" y="135" text-anchor="middle" class="qt">512 维特征</text>
<rect class="bx" x="536" y="70" width="104" height="40" rx="8"/><text x="588" y="95" text-anchor="middle" class="nm">1:1 验证</text>
<rect class="bx" x="536" y="130" width="104" height="40" rx="8"/><text x="588" y="155" text-anchor="middle" class="nm">1:N 检索</text>
<rect class="bx" x="262" y="185" width="100" height="44" rx="8"/><text x="312" y="205" text-anchor="middle" class="nm">密集关键点</text><text x="312" y="221" text-anchor="middle" class="qt">106 / 478 点</text>
<rect class="bx" x="394" y="185" width="110" height="44" rx="8"/><text x="449" y="205" text-anchor="middle" class="nm">特效</text><text x="449" y="221" text-anchor="middle" class="qt">贴纸·美颜·美妆</text>
<rect class="bx" x="262" y="20" width="100" height="44" rx="8"/><text x="312" y="40" text-anchor="middle" class="nm">3D 重建</text><text x="312" y="56" text-anchor="middle" class="qt">3DMM 62 参数</text>
<rect class="bx" x="394" y="20" width="110" height="44" rx="8"/><text x="449" y="40" text-anchor="middle" class="nm">属性编辑</text><text x="449" y="56" text-anchor="middle" class="qt">StarGAN</text>
<line class="ln" x1="98" y1="120" x2="130" y2="120"/><line class="ln" x1="230" y1="120" x2="262" y2="120"/><line class="ln" x1="362" y1="120" x2="394" y2="120"/>
<line class="ln" x1="504" y1="112" x2="536" y2="92"/><line class="ln" x1="504" y1="128" x2="536" y2="150"/>
<line class="ln" x1="200" y1="145" x2="262" y2="205"/><line class="ln" x1="362" y1="207" x2="394" y2="207"/>
<line class="ln" x1="200" y1="95" x2="262" y2="42"/><line class="ln" x1="230" y1="100" x2="394" y2="42"/>
<text x="680" y="120" class="qt">Gradio 界面</text><text x="680" y="137" class="qt">实时摄像头</text>
</svg></div><figcaption><b>一切从检测开始。</b>紫色三步是识别主干；关键点同时喂给特效模块，检测框同时喂给 3D 重建和属性编辑。</figcaption></figure>

### 专题索引 {#home-index}

<div class="scroll"><table class="idx"><thead><tr><th>专题</th><th>讲什么</th><th>对应代码</th></tr></thead><tbody>
<tr><td>一</td><td><a href="#t1">环境、工具与数据集</a></td><td><code>tools/env_check.py</code> <code>tools/explore_datasets.py</code></td></tr>
<tr><td>二</td><td><a href="#t2">深度学习基础</a></td><td><code>facevision/training/backbones.py</code></td></tr>
<tr><td>三</td><td><a href="#t3">人脸检测 SCRFD</a></td><td><code>facevision/detector.py</code></td></tr>
<tr><td>四</td><td><a href="#t4">关键点与对齐</a></td><td><code>facevision/align.py</code> <code>landmark.py</code></td></tr>
<tr><td>五</td><td><a href="#t5">识别与损失函数</a></td><td><code>facevision/training/losses.py</code></td></tr>
<tr><td>六</td><td><a href="#t6">训练与知识蒸馏</a></td><td><code>tools/train_recognition.py</code></td></tr>
<tr><td>七</td><td><a href="#t7">评测指标与结果</a></td><td><code>facevision/verification.py</code> <code>tools/eval_lfw.py</code></td></tr>
<tr><td>八</td><td><a href="#t8">量化、剪枝与部署</a></td><td><code>facevision/deploy.py</code></td></tr>
<tr><td>九</td><td><a href="#t9">实时特效</a></td><td><code>facevision/effects/</code></td></tr>
<tr><td>十</td><td><a href="#t10">3D 人脸重建</a></td><td><code>facevision/face3d/</code></td></tr>
<tr><td>十一</td><td><a href="#t11">GAN 属性编辑</a></td><td><code>facevision/gan/</code></td></tr>
<tr><td>十二</td><td><a href="#t12">系统集成</a></td><td><code>app/gradio_app.py</code></td></tr>
</tbody></table></div>

### 学习路线建议 {#home-route}

1. 先跑起来：按仓库 README 装环境、下载模型，执行 `python app/gradio_app.py`，在网页里把每个功能点一遍，知道"成品长什么样"。
2. 按专题一 → 五的顺序读，每读完一个专题就去打开对应的代码文件，对照讲义里的"代码精读"一行行看。
3. 做例题：先自己算，再看答案。例题里的数字都能在代码里打印出来验证。
4. 改代码做实验：例如把 `--head arcface` 换成 `cosface` 或 `softmax` 训练、把 NMS 阈值从 0.4 改成 0.2 看合影检测的变化、把量化校准图片从 512 张减到 16 张看精度。
5. 最后背必背清单，能脱稿讲清楚"一张照片进来，系统每一步在做什么、为什么这样做"。
