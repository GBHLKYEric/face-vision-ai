## 专题八|模型优化与部署|BN 融合、INT8 量化、剪枝、ONNX 与推理引擎 {#t8|八 优化·部署}

> 训练出好模型只是一半，另一半是让它在手机、普通电脑上跑得快、占得少。本章把"推理引擎（ByteNN、TensorRT、MNN）"里最常用的几种加速手段各实现一遍，并实测大小、速度、精度三方面的变化。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t8-k1">1 ONNX</a><a href="#t8-k2">2 层融合</a><a href="#t8-k3">3 量化原理</a><a href="#t8-k4">4 动态与静态量化</a><a href="#t8-k5">5 剪枝</a><a href="#t8-k6">6 蒸馏</a><a href="#t8-k7">7 移动端部署</a></dd>
<dt>实测</dt><dd><a href="#t8-r1">对比表</a></dd>
<dt>例题</dt><dd><a href="#t8-e1">BN 融合手算</a><a href="#t8-e2">量化手算</a></dd>
</dl></nav>

### 知识要点 {#t8-s1}

**1. ONNX：模型的"通用插头"。** PyTorch 模型导出成 ONNX（一张由 Conv、Add、Relu 等标准算子组成的计算图 + 权重），之后 ONNX Runtime（电脑）、TensorRT（NVIDIA）、MNN / NCNN / ByteNN（手机）都能读。导出后第一件事是和 PyTorch 输出逐元素对拍。 {#t8-k1}

**2. 层融合：Conv + BN 合二为一。** 推理时 BN 是一个固定的逐通道线性变换，可以直接并进前面卷积的权重和偏置，少一次读写内存，结果完全不变： {#t8-k2}

```latex
\mathrm{BN}(Wx+b)=\underbrace{W\cdot k}_{W'}\,x+\underbrace{(b-\mu)\,k+\beta}_{b'},\qquad k=\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}
```

`facevision/deploy.py` 的 `fuse_model` 手写了这个变换，单元测试验证融合前后输出一致。ONNX Runtime 在加载模型时也会自动做这件事（图优化级别 ORT_ENABLE_ALL）。

**3. 量化：用 8 位整数代替 32 位浮点。** 把一层的浮点数范围 [min, max] 均匀切成 256 份，每个浮点数用最近的那一格的编号（−128～127）表示： {#t8-k3}

```latex
\text{scale}=\frac{\max-\min}{255},\qquad q=\operatorname{round}\!\Bigl(\frac{x}{\text{scale}}\Bigr)+zp,\qquad \hat x=(q-zp)\cdot\text{scale}
```

模型小 4 倍；CPU 上 INT8 乘加指令（VNNI）一次能算的数是 FP32 的 4 倍。误差最多半格，神经网络对这种小扰动很鲁棒。权重"逐通道"量化（每个输出通道一个 scale）比整层一个 scale 精确得多。

**4. 动态量化 vs 静态量化。** 动态量化：权重提前量化好，激活值的范围在运行时每次现算——简单、不需要数据，但 PyTorch 只支持全连接层和 RNN，对卷积网络几乎没加速。静态量化：先用几百张真实图片（校准集）跑一遍，统计每层激活值的范围并固定下来，卷积也能全部用 INT8 算。本项目用 512 张 CelebA 人脸校准，范围取 99.99 百分位（去掉极少数离群值，否则一个异常大的值会把 scale 撑大、其余值精度变差）。 {#t8-k4}

**5. 剪枝。** 把绝对值最小的一部分权重置 0。非结构化剪枝（单个权重）可以剪掉很大比例而精度几乎不变，说明网络冗余很大：本项目全局剪掉 70% 的权重、不做微调，LFW 只从 98.867% 降到 98.633%（全局按绝对值排序时，全连接层被剪掉 87%，卷积层 50%）。此时特征本身已经变了（与原模型的余弦约 0.89），但 LFW 只看两张脸之间的相对距离，所以影响不大。不过稀疏矩阵在普通硬件上仍按稠密计算，大小和速度都不变。要真正提速需要结构化剪枝（整个卷积核、整个通道一起删）后重新微调。 {#t8-k5}

**6. 蒸馏也是压缩。** 专题六的学生 ResNet18 就是用 R50 老师蒸馏出来的：ONNX 单张延迟 12.5 ms，老师是 48.2 ms，快了约 3.9 倍，代价是 LFW 少了约 1 个百分点。剪枝、量化、蒸馏三者可以叠加使用。 {#t8-k6}

**7. 移动端部署的一般流程。** PyTorch 训练 → 导出 ONNX → 用推理引擎（MNN / NCNN / ByteNN / Core ML / TFLite）转换并量化 → 端侧 SDK 封装（相机采集 → 检测 → 跟踪 → 特效渲染，用 GPU/NPU 加速）。实时特效通常每隔几帧才做一次完整检测，中间帧只跟踪关键点，这是满足 30 FPS 的关键。 {#t8-k7}

### 实测 {#t8-s2}

#### 实测|大小 / 速度 / 精度对比 {#t8-r1}

{{DEPLOY_TABLE}}

### 例题 {#t8-s3}

#### 例题 1|BN 融合手算 {#t8-e1}

**一个 1×1 卷积权重 W = 2、偏置 b = 0，后接 BN：γ = 1.5，β = 0.2，μ = 1，σ² = 4（ε 忽略）。求融合后的 W′、b′，并用 x = 3 验算。**

```latex
\begin{aligned}
k&=1.5/\sqrt4=0.75,\qquad W'=2\times0.75=\boxed{1.5},\qquad b'=(0-1)\times0.75+0.2=\boxed{-0.55}\\
\text{融合前: }&1.5\times\frac{2\times3-1}{2}+0.2=3.95,\qquad \text{融合后: }1.5\times3-0.55=3.95
\end{aligned}
```

#### 例题 2|INT8 量化手算 {#t8-e2}

**某层激活值范围 [−2, 6]，用非对称 INT8 量化。求 scale、零点 zp，以及 x = 1.0 量化再反量化后的值。**

```latex
\begin{aligned}
\text{scale}&=\frac{6-(-2)}{255}=0.031373\\
zp&=-128-\operatorname{round}\!\Bigl(\frac{-2}{0.031373}\Bigr)=-128+64=-64\\
q&=\operatorname{round}(1.0/0.031373)+(-64)=32-64=-32\\
\hat x&=(-32+64)\times0.031373=\boxed{1.0039}\qquad\text{误差 }0.0039<\tfrac12\text{scale}
\end{aligned}
```
