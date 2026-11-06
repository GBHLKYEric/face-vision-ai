## 专题一|环境、工具与数据|Python 环境、图像在计算机里的样子、OpenCV、Docker / Git、三个数据集 {#t1|一 环境·数据}

> 阶段一的目标是把"能跑起来"这件事做扎实：环境可复现、数据看明白、第一个 OpenCV 程序跑通。这一章也解释了本项目在一台没有 NVIDIA 显卡的笔记本上，怎样用 Intel 核显完成训练。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t1-k1">1 软件栈</a><a href="#t1-k2">2 硬件与加速</a><a href="#t1-k3">3 图像是数组</a><a href="#t1-k4">4 OpenCV 基础</a><a href="#t1-k5">5 Docker 与 Git</a><a href="#t1-k6">6 三个数据集</a><a href="#t1-k7">7 数据许可</a></dd>
<dt>代码精读</dt><dd><a href="#t1-c1">opencv_basics.py</a><a href="#t1-c2">prepare_celeba.py</a></dd>
<dt>例题</dt><dd><a href="#t1-e1">内存映射</a><a href="#t1-e2">长路径坑</a></dd>
</dl></nav>

### 知识要点 {#t1-s1}

**1. 软件栈。** PyTorch：写模型、训练模型（动态图，调试方便）。ONNX：模型的"通用格式"，训练框架导出、各种推理引擎读入，一份模型可以跑在 Windows、Linux、手机、浏览器上。ONNX Runtime：微软的推理引擎，CPU 上跑 ONNX 模型很快，本项目的检测、关键点、识别推理都用它。OpenCV：图像读写、几何变换、滤波、画图。MediaPipe：谷歌的实时人脸网格。Gradio：几十行代码搭一个网页界面。版本和硬件信息见仓库里的 `docs/ENV_REPORT.md`（由 `tools/env_check.py` 自动生成）。 {#t1-k1}

**2. 没有 NVIDIA 显卡怎么训练。** 本机是 Intel Core Ultra 5 225H + Arc 130T 核显。PyTorch 从 2.5 起原生支持 Intel GPU，设备名叫 `xpu`（用法和 `cuda` 一样：`model.to("xpu")`）。实测核显 fp32 矩阵乘约 2.4 TFLOPS，bf16 训练 ResNet18 约 152 张/秒，是 CPU 的数倍。推理侧则用 ONNX Runtime 跑 CPU，对部署更有代表性（用户的手机、电脑大多没有独立显卡）。 {#t1-k2}

**3. 图像就是数字数组。** 一张 250×250 的彩色照片在 NumPy 里是形状 (250, 250, 3) 的 uint8 数组：第一维是行（高）、第二维是列（宽）、第三维是颜色通道，每个数 0～255。OpenCV 的通道顺序是 BGR（历史原因），PIL / 网络通常要 RGB。神经网络要的是 (N, C, H, W) 的 float 张量：批大小、通道、高、宽。整条流水线最常见的 bug 都出在这三处转换上。 {#t1-k3}

**4. OpenCV 基础操作。** 灰度化按人眼对颜色的敏感度加权（绿色权重最大）；高斯模糊用高斯核做加权平均去噪；Canny 边缘检测 = 求梯度 + 非极大值抑制 + 双阈值连接；直方图均衡把亮度分布拉平、提升暗光照片的对比度；仿射变换 `warpAffine` 是对齐、贴纸的基础。 {#t1-k4}

```latex
Y=0.299R+0.587G+0.114B
```

**5. Docker 与 Git。** Docker 把"操作系统 + Python + 所有依赖"打包成镜像，别人一条命令就能得到完全相同的环境（仓库里的 `Dockerfile`，`docker run --rm face-vision-ai` 运行 Hello World）。Git 记录每一次修改，数据集和模型权重太大不进仓库（`.gitignore`），而是用脚本 `scripts/download_assets.py` 重新下载——这也是工业界的标准做法。 {#t1-k5}

**6. 三个数据集。** {#t1-k6}

<div class="scroll"><table class="wide"><thead><tr><th>数据集</th><th>规模</th><th>用途</th><th>特点</th></tr></thead><tbody>
<tr><td>LFW</td><td>13,233 张 / 5,749 人</td><td>识别评测（6000 对，10 折）</td><td>长尾：4,069 人只有 1 张图；250×250，主角居中</td></tr>
<tr><td>CelebA</td><td>202,599 张 / 10,177 人 / 40 个属性</td><td>识别训练、StarGAN 训练与 FID 评测</td><td>人物与 LFW 不重叠；属性极不均衡（如秃头只占 2%）</td></tr>
<tr><td>WIDER FACE</td><td>验证集 3,226 张 / 39,708 张脸</td><td>检测评测（Easy/Medium/Hard）</td><td>大量 10 像素级小脸、遮挡、合影，见专题三</td></tr>
</tbody></table></div>

PDF 里提到的 MS-Celeb-1M 已于 2019 年因隐私问题被微软下架，本项目改用许可更清晰、并且和 LFW 人物不重叠的 CelebA 训练——"训练集与测试集人物不重叠"是人脸识别实验最基本的要求，否则准确率是虚高的。分析脚本是 `tools/explore_datasets.py`，图表在 `outputs/datasets/`。

**7. 人脸数据的许可与隐私。** 这些学术数据集只能用于非商业研究；人脸属于敏感个人信息，真实产品必须取得用户授权、本地处理或加密传输。项目代码里所有演示都用公开数据集里的公众人物照片。 {#t1-k7}

### 代码精读 {#t1-s2}

#### 代码 1|tools/opencv_basics.py：第一个 OpenCV 程序 {#t1-c1}

```python
img = cv2.imread(path)                          # BGR, (高, 宽, 3), uint8
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度
blur = cv2.GaussianBlur(img, (9, 9), 0)         # 高斯模糊
edges = cv2.Canny(gray, 80, 160)                # 边缘
eq = cv2.equalizeHist(gray)                     # 直方图均衡
boxes = cascade.detectMultiScale(gray, 1.1, 5)  # 2001 年的 Haar 人脸检测, 用来和深度学习对比
```

易错点：OpenCV 5 不再自带 Haar 级联的 xml 文件，`cv2.data.haarcascades` 路径下是空的。项目把它加进了下载脚本，这类"版本升级后静默失效"的问题要靠读报错信息定位。

#### 代码 2|tools/prepare_celeba.py：20 万张图的预处理 {#t1-c2}

每张 CelebA 图：PNG 解码 → SCRFD 检测 5 个点 → 与 LFW 评测完全相同的对齐 → 112×112，结果存进一个 (202599, 112, 112, 3) 的大数组（7.6 GB）。"训练和测试用同一套对齐"非常重要：如果训练时用 CelebA 自带的粗对齐、测试时用 SCRFD 对齐，两者眼睛位置差几个像素，准确率就会掉。

第一版用 6 个 CPU 进程，每张约 360 ms，算下来要 2.5 小时；改成多线程解码 + 核显批量检测后每分片约 100 秒。后来才查到慢的另一半原因：Windows 11 会把没有窗口的后台进程调到能效核上（EcoQoS），这种进程跑约 2 秒后就会变慢 3 倍左右，所以 `facevision/utils.py` 里有一个只对本进程关闭节流的函数。性能优化的第一步永远是先测量瓶颈在哪里，而不是凭感觉改。

### 例题 {#t1-s3}

#### 例题 1|7.6 GB 的数组怎么高效地喂给训练 {#t1-e1}

**训练时每个 batch 随机取 128 张脸。是把 7.6 GB 全读进内存，还是每次从磁盘读？**

用内存映射：`np.load(path, mmap_mode="r")`。数组"看起来"整个在内存里，实际上访问到哪一部分，操作系统才从磁盘读那一部分，并自动缓存。第一轮稍慢，之后几乎就是读内存的速度。每个 batch 用花式索引取出 128 张（约 4.8 MB），搬到核显上再归一化和翻转，比 PyTorch DataLoader 多进程解码 JPEG 简单得多。

```latex
128\times112\times112\times3\ \text{字节}=4{,}816{,}896\ \text{字节}\approx\boxed{4.8\ \text{MB}/\text{batch}}
```

#### 例题 2|Windows 路径太长 {#t1-e2}

**项目最初放在一个很深的目录里，pip 安装包和解压 LFW 时报"找不到文件"，为什么？**

Windows 默认限制完整路径不超过 260 个字符。工作目录本身已经 150 多个字符，再加上 `lfw/Mireya_Elisa_Moscoso_Rodriguez/Mireya_Elisa_Moscoso_Rodriguez_0001.jpg` 就超了。解决办法：把项目放到短路径（`D:\face-vision-ai`），虚拟环境放到 `C:\Users\0\.venvs\facevision`。开启系统的长路径支持也行，但那属于修改系统设置。

结论：环境问题往往不报"真正的原因"，而是报一个看起来无关的错误（这里是 "No such file or directory"）。养成先看路径、版本、编码的习惯。
