## 专题十二|系统集成：把所有模块拼成一个应用|Gradio 界面、实时视频流水线、懒加载、性能分析 {#t12|十二 系统集成}

> 阶段四的目标：把识别和特效整合成一个有界面、能接摄像头的系统，并在 CPU 上达到实时。本章讲 `app/gradio_app.py` 和 `tools/realtime_demo.py` 的设计。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t12-k1">1 系统结构</a><a href="#t12-k2">2 懒加载</a><a href="#t12-k3">3 实时视频</a><a href="#t12-k4">4 性能预算</a><a href="#t12-k5">5 可复现</a></dd>
<dt>代码精读</dt><dd><a href="#t12-c1">gradio_app.py</a></dd>
<dt>例题</dt><dd><a href="#t12-e1">帧率预算</a></dd>
</dl></nav>

### 知识要点 {#t12-s1}

**1. 系统结构。** `facevision/` 是算法库（检测、对齐、识别、特效、3D、GAN 各自独立，互相只通过 numpy 数组和简单的数据类交换数据）；`tools/` 是命令行脚本（评测、训练、量化、演示）；`app/` 是界面。界面只调用库，不包含算法——这样同一套算法既能被网页用，也能被命令行、单元测试、以后的手机 SDK 用。 {#t12-k1}

**2. 懒加载与缓存。** 界面启动时一个模型都不加载，第一次点到某个功能才加载对应模型（`functools.cache` 保证只加载一次）。启动快、内存省，也避免一个模块的依赖问题拖垮整个界面。 {#t12-k2}

**3. 实时视频的流水线。** 摄像头每帧：MediaPipe 检测 478 个点 → One Euro 滤波去抖 → 按勾选顺序叠加特效 → 显示。Gradio 的摄像头组件以流的方式每 50 ms 发一帧到后端；`tools/realtime_demo.py` 则直接用 OpenCV 读摄像头，延迟更低，适合测帧率。 {#t12-k3}

**4. 性能预算。** 30 FPS 意味着每帧只有 33 ms。要把这 33 ms 分给"检测/关键点 + 每个特效 + 编码显示"，任何一步超时都会掉帧。实测数据见下面的例题和专题九。 {#t12-k4}

易错点：模块单独测得快，拼起来未必快。本项目的 5 个 ONNX 模型单独测加起来约 200 ms，串成流水线却要 2122 ms。原因是 ONNX Runtime 的线程干完活后默认空转等下一个任务，5 个模型各有一组线程在空转，互相抢 CPU。在 `facevision/onnx_utils.py` 里关掉空转（`session.intra_op.allow_spinning = 0`）后降到 99 ms，快了 21 倍，单个模型的速度不受影响。所以性能要测端到端，分项数字加起来对不上时一定要追查。

**5. 可复现。** `scripts/download_assets.py` 一键下载所有模型和数据集；`tools/env_check.py` 生成环境报告；`Dockerfile` 固定运行环境；每个评测脚本把结果写成 JSON。任何人拿到仓库都能得到同样的数字。 {#t12-k5}

### 代码精读 {#t12-s2}

#### 代码 1|app/gradio_app.py：一个标签页的完整写法 {#t12-c1}

```python
@functools.cache
def analyzer() -> FaceAnalyzer:                 # 懒加载 + 只加载一次
    return FaceAnalyzer(modules=("landmark_106", "recognition", "genderage"))

def verify_tab(img1, img2):
    embs = []
    for img in (img1, img2):
        faces = analyzer().analyze(rgb2bgr(img), max_num=1)    # Gradio 给的是 RGB
        embs.append(faces[0].embedding)
    sim = float(np.dot(embs[0], embs[1]))                       # 已归一化, 点积 = 余弦
    return ("同一个人" if sim > THRESH else "不是同一个人"), sim

with gr.Tab("1:1 人脸验证"):
    v1, v2 = gr.Image(label="照片 A"), gr.Image(label="照片 B")
    gr.Button("比对").click(verify_tab, [v1, v2], [out_img, out_txt])
```

阈值 THRESH 直接读 LFW 10 折评测得到的平均阈值，而不是拍脑袋写一个数。

### 例题 {#t12-s3}

#### 例题 1|实时帧率预算 {#t12-e1}

{{SYSTEM_PERF}}
