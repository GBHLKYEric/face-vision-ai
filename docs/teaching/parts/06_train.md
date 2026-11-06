## 专题六|自己训练识别模型|CelebA + IResNet18 + ArcFace + 知识蒸馏，在 Intel 核显上训练 {#t6|六 训练·蒸馏}

> 专题五讲了损失函数，这一章把它真正跑起来：在一台没有独立显卡的笔记本上，用 20 万张 CelebA 人脸从零训练一个 ResNet18 识别模型，并用 R50 大模型当老师做知识蒸馏，最后在 LFW 上按官方协议测准确率。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t6-k1">1 训练方案</a><a href="#t6-k2">2 知识蒸馏</a><a href="#t6-k3">3 老师特征提取</a><a href="#t6-k4">4 训练循环</a><a href="#t6-k5">5 训练曲线怎么看</a></dd>
<dt>代码精读</dt><dd><a href="#t6-c1">train_recognition.py</a><a href="#t6-c2">extract_teacher.py</a></dd>
<dt>实验结果</dt><dd><a href="#t6-r1">训练曲线与最终结果</a></dd>
<dt>例题</dt><dd><a href="#t6-e1">每轮多少步</a><a href="#t6-e2">蒸馏损失的量级</a></dd>
</dl></nav>

### 知识要点 {#t6-s1}

**1. 训练方案与取舍。** PDF 要求在 MS-Celeb-1M 子集上训练 ResNet50 + ArcFace，达到 LFW 98.5% 以上。实际约束：MS-Celeb-1M 已下架；本机只有 Intel 核显，实测 ResNet50 训练 86 张/秒、ResNet18 152 张/秒。最终方案：CelebA 全部 202,599 张 / 10,177 人（与 LFW 人物不重叠）+ IResNet18 + ArcFace（s = 64，m = 0.5）+ 来自 R50 老师的特征蒸馏，SGD 动量 0.9、学习率 0.05、批大小 128、bf16 混合精度、1 轮预热 + 余弦退火。 {#t6-k1}

```latex
L=\underbrace{L_{\text{ArcFace}}(f_s,\,y)}_{\text{身份标签}}+\lambda\cdot\underbrace{\bigl(1-\cos(f_s,\,f_t)\bigr)}_{\text{模仿老师}},\qquad \lambda=50
```

**2. 知识蒸馏：让小模型模仿大模型。** 老师（ArcFace-R50，在约 60 万人的 WebFace600K 上训练，LFW 99.85%）已经学会了"什么样的两张脸是同一个人"。学生每看到一张脸，除了要猜对身份（ArcFace 损失），还要让自己的 512 维特征方向尽量对齐老师的特征方向。老师的特征里藏着标签没有的信息——"这个人和那个人长得有多像"（暗知识），而且老师见过的人比 CelebA 多 60 倍。小数据集上，蒸馏能让学生学得更快、更好。 {#t6-k2}

类比：自学（只看标准答案）和跟着名师学（看名师的解题思路）。名师的思路里有"这道题和那道题是一类"的信息，标准答案里没有。

**3. 老师特征提前算好。** 训练 10 轮要看 200 万次图，每次都跑一遍 R50 老师太浪费。由于数据增强只有水平翻转，而老师特征本身就是"原图 + 翻转图"相加得到的、对翻转近似不变，所以每张图的老师特征只需要提前算一次，存成 (202599, 512) 的 float16 数组（207 MB），训练时直接读。 {#t6-k3}

**4. 训练循环五步。** 每个 batch：① 取 128 张脸和对应的身份标签、老师特征；② 前向得到学生特征；③ 算 ArcFace 损失 + 蒸馏损失；④ 反向传播得到梯度，梯度范数超过 5 就等比缩小（梯度裁剪，防止偶发的大梯度把训练搞崩）；⑤ SGD 更新参数。每轮结束在 LFW 6000 对上测一次准确率，画曲线。 {#t6-k4}

**5. 怎么读训练曲线。** 损失应该整体下降；训练集分类准确率（这一批里 cos 最大的类是不是正确类）应该上升；蒸馏损失下降说明学生越来越像老师；LFW 准确率是最终目标。如果训练集准确率很高、LFW 却不涨，就是过拟合（CelebA 每人平均只有 20 张，容易发生）。 {#t6-k5}

### 代码精读 {#t6-s2}

#### 代码 1|tools/train_recognition.py：核心循环 {#t6-c1}

```python
for imgs, y, t_feat in batches:                       # 内存映射随机取 batch
    lr = lr_at(step, total_steps, warmup, args.lr)     # 预热 + 余弦
    x = to_tensor(imgs, dev)                           # uint8 BGR -> [-1,1] RGB, 搬到 xpu
    with torch.autocast(device_type=dev, dtype=torch.bfloat16):
        emb = model(x)                                 # bf16 前向, 快一倍
    emb = emb.float()                                  # 损失用 fp32 算 (acos 对精度敏感)
    loss_margin, cos = head(emb, y)                    # ArcFace
    loss_kd = distillation_loss(emb, teacher_feat)     # 1 - cos(学生, 老师)
    loss = loss_margin + args.kd_weight * loss_kd
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)        # 梯度裁剪
    opt.step()
```

两个细节：BN 和 PReLU 的参数单独分组、不做权重衰减；保存最终模型时只存骨干网络（`final.pt`），10177 类的分类头部署时用不到。

#### 代码 2|tools/extract_teacher.py：先对拍再批量 {#t6-c2}

```python
model = convert(onnx.load(str(REC_MODEL))).eval().to("xpu")     # ONNX -> PyTorch -> 核显
ref = ArcFaceONNX().embed(sample, flip_test=False)              # 官方 ONNX Runtime 结果
mine = F.normalize(model(to_tensor(sample, "xpu")), dim=1)      # 转换后的结果
assert ((ref * mine).sum(1)).min() > 0.9999                     # 余弦相似度必须几乎为 1
```

先用 8 张图比对转换前后输出一致，再批量提取。这一步看似多余，却能拦住专题三里那种"batch 维顺序不对"的静默错误。

### 实验结果 {#t6-s3}

#### 结果 1|训练曲线与最终结果 {#t6-r1}

{{TRAIN_RESULT}}

### 例题 {#t6-s4}

#### 例题 1|每轮多少步、要多久 {#t6-e1}

**202,599 张图、批大小 128、核显约 150 张/秒，训练 10 轮，每轮多少步？总共大约多久？**

```latex
\begin{aligned}
\text{每轮步数}&=\lfloor 202599/128\rfloor=1582\\
\text{每轮时间}&\approx 202599/150\approx 1351\ \text{s}\approx 22.5\ \text{min}\\
\text{10 轮}&\approx\boxed{3.75\ \text{小时}}
\end{aligned}
```

实测每轮约 22 分钟，和估算一致。数据中心级 GPU 的吞吐量比核显高一到两个数量级，同样的训练可以从几小时缩到几十分钟以内。硬件决定了实验迭代的速度，这也是 PDF 里提到 BytePS 分布式训练的原因。

#### 例题 2|为什么蒸馏权重取 50 {#t6-e2}

**训练中段，ArcFace 损失约 10，学生和老师特征的余弦约 0.9。λ 取 1 和取 50 时，两项损失分别多大？**

蒸馏损失 = 1 − 0.9 = 0.1。λ = 1 时蒸馏项只有 0.1，相比 ArcFace 的 10 可以忽略，等于没蒸馏；λ = 50 时蒸馏项为 5，两项量级相当，学生同时听"标签"和"老师"的话。

```latex
\lambda=1:\ 10+0.1,\qquad \lambda=50:\ 10+\boxed{5}
```

易错点：加权损失时要看各项的量级，而不是只看公式。ArcFace 的 logit 乘了 s = 64，它的损失天然比余弦距离大两个数量级。
