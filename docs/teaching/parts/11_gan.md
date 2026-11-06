## 专题十一|GAN 人脸属性编辑：一个生成器改发色、性别、年龄|生成器与判别器的博弈、WGAN-GP、StarGAN 的三个损失、FID，以及扩散模型 {#t11|十一 GAN 编辑}

> 前面的专题都在"看懂"人脸：检测、对齐、识别、估计性别年龄。这一章反过来"画"人脸：给一张照片，把头发变成金色、把男人变成女人、把年轻人变老。本项目用 StarGAN（CVPR 2018）的官方 CelebA 权重，在 `facevision/gan/stargan.py` 里按官方结构重写了网络（能 strict 加载），补上了"任意照片 → 对齐 → 编辑 → 贴回"的流程；训练代码在 `tools/train_stargan.py`，评测在 `tools/eval_stargan.py`。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t11-k1">1 GAN 博弈</a><a href="#t11-k2">2 DCGAN 到 StyleGAN</a><a href="#t11-k3">3 条件生成</a><a href="#t11-k4">4 WGAN-GP</a><a href="#t11-k5">5 三个损失</a><a href="#t11-k6">6 循环一致</a><a href="#t11-k7">7 CelebA 预处理</a><a href="#t11-k8">8 贴回原图</a><a href="#t11-k9">9 FID 与实测</a><a href="#t11-k10">10 扩散模型</a></dd>
<dt>代码精读</dt><dd><a href="#t11-c1">条件输入</a><a href="#t11-c2">训练一步</a><a href="#t11-c3">InstanceNorm 陷阱</a><a href="#t11-c4">贴回原图</a><a href="#t11-c5">自己算 FID</a></dd>
<dt>例题</dt><dd><a href="#t11-e1">FID 手算</a><a href="#t11-e2">初始损失</a><a href="#t11-e3">标签向量</a><a href="#t11-e4">坐标换算</a></dd>
</dl></nav>

### 知识要点 {#t11-s1}

**1. GAN：两个网络的博弈。** 生成对抗网络（GAN，Goodfellow 2014）由两个网络组成：生成器 G 负责"造图"，判别器 D 负责"验图"——给它一张图，它判断这是真实照片还是 G 造的。训练时两者交替更新：D 努力把真图和假图分开，G 努力让 D 分不开。博弈的平衡点是 G 造出的图在统计意义上和真图无法区分。原始 GAN 的目标函数： {#t11-k1}

```latex
\min_G\max_D\;\mathbb{E}_{x\sim p_{\text{data}}}\big[\log D(x)\big]+\mathbb{E}_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]
```

这里 D(x) 是"x 为真"的概率，z 是随机噪声。注意 G 从来没见过真图本身，它学到的一切都来自 D 回传的梯度："你这张图哪里不像真的"。

类比：造假币的人（G）和验钞机（D）。验钞机越来越严，逼着造假的人把水印、纹路做得越来越像；最后验钞机只能靠猜（判对率 50%）。

**2. 从 DCGAN 到 StyleGAN。** DCGAN（2015）确立了"全卷积 + BatchNorm + 转置卷积上采样"的结构，第一次稳定生成 64×64 的人脸。WGAN / WGAN-GP（2017）换了损失函数，解决"训练一会儿就崩"的问题（要点 4）。Pix2Pix（2017）做成对图像翻译（草图 → 照片），CycleGAN（2017）用循环一致做不成对的翻译（马 ↔ 斑马）。ProGAN（2018）从 4×4 逐级加层训练到 1024×1024。StarGAN（2018）用一个生成器完成多个域之间的翻译，就是本章的主角。StyleGAN / StyleGAN2（2019/2020）用映射网络把噪声变成"风格向量"，在每一层调制特征，生成 1024×1024 以假乱真的人脸，而且潜空间里不同方向对应年龄、表情、朝向，可以直接拿来编辑。 {#t11-k2}

**3. 条件生成：把标签拼成额外的通道。** 普通 GAN 的 G 只吃噪声，生成什么全凭运气。StarGAN 的 G 吃两样东西：一张真实人脸 x，和一个"目标属性"向量 c（5 维 0/1：黑发、金发、棕发、男性、年轻）。把 c 的每一维在空间上复制成一张 128×128 的常数图，和 RGB 三个通道拼在一起，网络第一层就看到 8 个通道： {#t11-k3}

```latex
G(x,c)=\mathrm{Net}\Big(\operatorname{concat}\big(x,\;\operatorname{tile}(c)\big)\Big),\qquad \operatorname{concat}\big(x,\operatorname{tile}(c)\big)\in\mathbb{R}^{(3+5)\times 128\times 128}
```

这个办法很朴素，但有一个隐藏条件：卷积必须用补零（zero padding）。标签通道是常数图，它对第一层卷积输出的贡献在图像内部处处相同，相当于一个偏置；只有在边缘几个像素，补零让这个贡献不再是常数。紧接着的 InstanceNorm 会减去每个通道的空间均值，内部那个"偏置"几乎被抹掉，标签信息主要靠边缘的补零效应传进网络。如果改用镜像补边，连边缘也是常数，标签就被完全抹掉，网络"看不见"标签（官方代码的注释里专门提醒了这一点）。

**4. WGAN-GP：让训练稳定下来。** 原始 GAN 的 D 输出概率。D 一旦太强，对假图的输出贴近 0，log(1 − D) 几乎是平的，G 拿不到梯度，训练就卡住或崩溃。WGAN 改为估计两个分布之间的 Wasserstein 距离（"把一堆土搬成另一堆土最少要做多少功"）。它的对偶形式要求 D（此时叫 critic，输出任意实数，不再是概率）是 1-Lipschitz 函数，即对输入的梯度范数不超过 1。WGAN-GP 的做法是在真图和假图之间随机插值得到 x̂，惩罚梯度范数偏离 1： {#t11-k4}

```latex
\begin{aligned}
W(p_r,p_g)&=\sup_{\lVert f\rVert_L\le 1}\;\mathbb{E}_{x\sim p_r}\big[f(x)\big]-\mathbb{E}_{\tilde x\sim p_g}\big[f(\tilde x)\big]\\
\mathcal{L}_{\text{gp}}&=\mathbb{E}_{\hat x}\Big[\big(\lVert\nabla_{\hat x}D_{\text{src}}(\hat x)\rVert_2-1\big)^2\Big],\qquad \hat x=\alpha x+(1-\alpha)\,G(x,c),\ \ \alpha\sim U(0,1)
\end{aligned}
```

为了这个梯度惩罚，D 里不能用 BatchNorm（它让一个样本的输出依赖同 batch 的其它样本，"对单个输入求梯度"就没有意义了），所以 StarGAN 的 D 只有卷积和 LeakyReLU。

**5. StarGAN 的三个损失。** D 有两个输出头：D_src 是真假分数（PatchGAN，输出 2×2 个分数，每个只看图像的一块）；D_cls 是 5 个属性的 logits（辅助分类器）。训练一步时，目标标签 c 取本 batch 真实标签随机打乱后的结果，保证属性组合都是真实存在的。两个网络的损失： {#t11-k5}

```latex
\begin{aligned}
\mathcal{L}_D&=\underbrace{-\mathbb{E}\big[D_{\text{src}}(x)\big]+\mathbb{E}\big[D_{\text{src}}(G(x,c))\big]}_{\text{Wasserstein}}+\lambda_{\text{gp}}\,\mathcal{L}_{\text{gp}}+\lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}^{r}\\
\mathcal{L}_G&=-\mathbb{E}\big[D_{\text{src}}(G(x,c))\big]+\lambda_{\text{cls}}\,\mathcal{L}_{\text{cls}}^{f}+\lambda_{\text{rec}}\,\mathcal{L}_{\text{rec}}\\
\lambda_{\text{gp}}&=10,\quad \lambda_{\text{cls}}=1,\quad \lambda_{\text{rec}}=10
\end{aligned}
```

分类损失是 5 个独立的二分类交叉熵之和（一张图可以同时是"金发 + 女性 + 年轻"）。D 只用真图训练分类头（学会"认属性"），G 则要求假图被 D 认成目标属性（真的把属性改到位）：

```latex
\mathcal{L}_{\text{cls}}^{r}=\sum_{i=1}^{5}\mathrm{BCE}\big(D_{\text{cls}}(x)_i,\;c'_i\big),\qquad
\mathcal{L}_{\text{cls}}^{f}=\sum_{i=1}^{5}\mathrm{BCE}\big(D_{\text{cls}}(G(x,c))_i,\;c_i\big)
```

其余超参数：每更新 5 次 D 才更新 1 次 G（n_critic = 5，critic 要先训准，梯度才有意义）；Adam，学习率 1e-4，β = (0.5, 0.999)；batch 16 共 20 万步，后 10 万步学习率线性降到 0。

易错点：官方 `solver.py` 每次衰减只减 lr / num_iters_decay，按默认参数训练结束时学习率只降了 0.1%，与论文"线性降到 0"不符。本项目的 `train_stargan.py` 按论文实现。

**6. 循环一致：只改属性，不改人。** 只有前两项时，G 有偷懒的办法：随便生成一张"金发美女"就能骗过 D 并通过分类，根本不需要像输入的那个人。循环一致损失要求"改过去再改回来"能还原原图： {#t11-k6}

```latex
\mathcal{L}_{\text{rec}}=\mathbb{E}_{x,c,c'}\Big[\big\lVert x-G\big(G(x,c),\,c'\big)\big\rVert_1\Big]
```

这里 c' 是原标签。只有保留了身份、姿态、背景的信息，才能还原回去。用 L1 而不是 L2，是因为 L2 对大误差惩罚过重，会让结果偏模糊。

类比：把一段中文翻译成英文再翻回中文，如果意思走样了，说明第一次翻译丢了信息。

**7. CelebA 预处理与对齐空间。** CelebA 有 20 万张名人照片，每张标注了 40 个属性（StarGAN 用其中 5 个），官方发布的"对齐版"已经按双眼位置把人脸摆正、裁成 178×218。StarGAN 官方预处理是： {#t11-k7}

```latex
178\times 218\xrightarrow{\ \text{CenterCrop}(178)\ }178\times 178\xrightarrow{\ \text{Resize}(128)\ }128\times 128\xrightarrow{\ x/127.5-1\ }[-1,\,1]
```

G 只认识"这样摆放"的脸。为了让任意照片也落进这个空间，我们用 SCRFD 检测 CelebA valid 划分前 400 张对齐图的 5 个关键点，取平均作为标准位置（左眼 (69.58, 111.59)、右眼 (108.26, 111.66)、鼻尖 (88.72, 135.73)、左右嘴角 (72.56, 152.72) / (105.80, 152.82)），与 CelebA 自带关键点文件的平均值几乎一致。

**8. 任意照片：对齐、编辑、贴回。** `StarGANEditor.edit_photo` 的四步：SCRFD 检测 5 点 → Umeyama 相似变换把 5 点对到标准位置，裁出 128×128 → G 编辑 → 用逆变换贴回原图。原属性 c' 从哪来？照片没有标注，我们复用 D 的辅助分类头来估计（它在训练中一直被要求"认出真图的属性"）。贴回时有两种做法： {#t11-k8}

```latex
\begin{aligned}
\text{replace:}\quad I_{\text{out}}&=(1-m)\odot I+m\odot\mathcal{W}_{A^{-1}}\big(G(x,c)\big)\\
\text{residual:}\quad I_{\text{out}}&=I+m\odot\mathcal{W}_{A^{-1}}\big(G(x,c)-x\big)
\end{aligned}
```

A 是"原图 → 128 输入"的仿射矩阵，W 表示按逆变换重采样，m 是羽化（高斯模糊边缘）的椭圆蒙版，覆盖头发和脸。replace 直接把放大后的 128×128 结果盖上去，人脸较大时会明显发糊；residual 只把"G 改了什么"放大后加回原图，原图的毛孔、发丝等高频细节都保留下来，本项目默认用它。

<figure class="fig"><img class="shot" src="img/gan_blend_compare.jpg" alt="LFW 照片上 residual 与 replace 两种贴回方式的对比"><figcaption><b>贴回方式对比。</b>第 2 列是对齐后送进网络的 128×128 输入；residual 列比 replace 列清晰，因为只叠加了"变化量"。</figcaption></figure>

**9. 怎么评价生成质量：FID 与 IS。** 生成图没有"标准答案"，只能比较分布。FID 把真图和生成图分别送进 ImageNet 上训练的 Inception-v3，取 2048 维特征，各拟合一个多元高斯 N(μ, Σ)，计算两个高斯之间的 Fréchet 距离，越小越好；IS（Inception Score）只看生成图：每张图分类越确定、整体类别越多样，分数越高，越大越好，但它不和真图比较，人脸这种"只有一类"的数据上意义不大。 {#t11-k9}

```latex
\mathrm{FID}=\lVert\mu_r-\mu_g\rVert_2^2+\operatorname{Tr}\Big(\Sigma_r+\Sigma_g-2\big(\Sigma_r\Sigma_g\big)^{1/2}\Big),\qquad
\mathrm{IS}=\exp\Big(\mathbb{E}_{x}\,D_{\mathrm{KL}}\big(p(y\mid x)\,\Vert\,p(y)\big)\Big)
```

本项目实测（`tools/eval_stargan.py`，官方权重）：CelebA 官方 test 划分的前 2000 张作为输入 A，接下来 2000 张作为真实参考集 B（parquet 按身份排列，A、B 基本是不同的人）；原标签 c' 用 CelebA 标注；"成功率"由独立的 InsightFace GenderAge 模型判定（它与 StarGAN 无关，在真图上与 CelebA 性别标注的一致率为 97.3%）。

<div class="scroll"><table class="wide">
<thead><tr><th>编辑</th><th>FID（vs 真图 B）</th><th>其它指标</th></tr></thead>
<tbody>
<tr><td>真图 A（基线）</td><td>16.72</td><td>同分布、同样 2000 张：这是 FID 在这个样本量下的"地板"</td></tr>
<tr><td>重建 G(x, c')</td><td>21.47</td><td>L1 = 0.092（[−1, 1] 尺度），即平均每像素差 11.7 个灰度级</td></tr>
<tr><td>黑发</td><td>29.29</td><td>与真实黑发照片比 26.90（输入本身与黑发照片比 27.49）</td></tr>
<tr><td>金发</td><td>33.90</td><td>与真实金发照片比 23.69（输入本身 27.47）</td></tr>
<tr><td>棕发</td><td>25.47</td><td>与真实棕发照片比 22.11（输入本身 17.52）</td></tr>
<tr><td>性别翻转</td><td>28.74</td><td>GenderAge 判定性别改变 86.7%（男→女 90.2%，女→男 85.6%）</td></tr>
<tr><td>年龄翻转</td><td>29.42</td><td>年轻→年长：预测年龄升高 99.6%，平均 +25.2 岁；年长→年轻：降低 92.7%，平均 −12.9 岁</td></tr>
</tbody></table></div>

怎么读这张表：(1) 所有编辑的 FID 都比基线高 9～17，说明编辑后的图仍然像人脸照片，但带有 128×128 生成图的模糊和伪影。(2) 金发对 B 的 FID 最高，但和"真实金发照片"比只有 23.69，比输入本身离金发分布还近——FID 高主要是因为"全部变成金发"偏离了"各种发色混在一起"的真实分布，而不是画质差。所以单看一个 FID 数字容易误判，要选对参考集。(3) 年长→年轻的平均变化只有反方向的一半，可能的原因之一是 CelebA 里"年长"样本少（这 2000 张里只有 358 张），G 在这个方向上练得不够。

易错点：官方 StarGAN 训练时只随机留出约 2000 张做测试，其余约 20 万张（包括 CelebA 官方 test 划分的绝大部分）都参与了训练，所以上面的重建误差和 FID 偏乐观，衡量的是"训练域内"的效果。另外 FID 只用了 2000 张，绝对值会偏高，只适合在相同设置下互相比较。GenderAge 预测的绝对年龄偏大（标注"年轻"的图平均预测 32.9 岁，"非年轻"49.9 岁），所以只看变化方向。D 的辅助分类器认为目标属性达成了 96%～99%，但 G 正是对着这个 D 训练的，这个数字偏乐观，独立模型的结果更可信。

<figure class="fig"><img class="shot" src="img/gan_celeba_grid.jpg" alt="CelebA test 图上的 5 个单属性编辑"><figcaption><b>CelebA test 图上的 5 个单属性编辑（输入 + 黑 / 金 / 棕发 + 性别翻转 + 年龄翻转）。</b>第 4 行是一张很暗、上方被文字占满的低质量图，编辑基本失败；第 1、7 行可以看到"变黑发"连带把肤色也调深了——CelebA 里发色和肤色、妆容高度相关，G 把它们一起学了进去。</figcaption></figure>

<figure class="fig"><img class="shot" src="img/gan_lfw_grid.jpg" alt="LFW 照片上的 edit_photo 结果"><figcaption><b>任意照片（LFW 250×250）上的 edit_photo。</b>原属性由 D 的分类头估计：5 个人的性别和年龄都估对了，但三种发色全部判为 0（白发、灰发本来就不在这三类里；Jolie 和 Serena 的深色头发也没有超过 0.5 的阈值），这时"变某种发色"仍然有效，只是没有"原发色"可以清零。年长→年轻的变化（第 1、3、5 行最后一列）明显弱于反方向；"变金发"时肤色也被调浅，头部周围偶尔能看到一圈淡淡的光晕（蒙版内背景被轻微改动）。</figcaption></figure>

CPU 速度（4 个线程，测量时 CPU 正被其它任务占用，偏慢）：G 处理一张 128×128 约 0.2 秒；一张 250×250 的照片做一个编辑（含检测、D 估计属性、G、贴回）约 0.76 秒，一次做 5 个编辑约 1.4 秒（5 个编辑拼成一个 batch，检测和对齐只做一次）。

**10. 扩散模型：为什么今天的主角换了。** 扩散模型（DDPM 2020、Stable Diffusion 2022）走的是另一条路：先定义一个"加噪"过程，把真图一步步加高斯噪声直到变成纯噪声；再训练一个网络学"去噪"——给它一张加了噪声的图和步数 t，预测加进去的噪声。生成时从纯噪声出发，反复去噪几十步得到图像： {#t11-k10}

```latex
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I),\qquad
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}\Big[\big\lVert\epsilon-\epsilon_\theta(x_t,t)\big\rVert_2^2\Big]
```

它在画质和多样性上超过 GAN，主要有三个原因：(1) 损失是普通的回归（预测噪声），没有两个网络的博弈，训练稳定，模型和数据可以一直放大；(2) 它要覆盖整个数据分布，不会像 GAN 那样"模式崩溃"（只会画几种讨好 D 的脸）；(3) 文本、草图、蒙版等条件都可以方便地接进去（classifier-free guidance），编辑能力强（如 SDEdit、InstructPix2Pix）。代价是慢：要几十次前向，GAN 只要一次。所以实时美颜、视频特效这类对速度敏感的场景至今还常用 GAN 或"蒸馏成一步"的扩散模型。

结论：GAN 的核心是"对抗"——用一个可学习的判别器代替手写的损失函数；StarGAN 在此之上加了"条件 + 分类 + 循环一致"，用一个网络完成多种属性编辑。

### 代码精读 {#t11-s2}

#### 代码 1|facevision/gan/stargan.py：标签怎么"告诉"生成器 {#t11-c1}

网络结构逐层照搬官方实现（层的顺序、参数名都一致），因此官方 `200000-G.ckpt` 能 `strict=True` 加载，一个参数都不多、不少。条件输入只有两行：

```python
def forward(self, x, c):
    # c: (B, 5) -> (B, 5, 1, 1) -> 复制成 (B, 5, H, W) 的常数图
    c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
    return self.main(torch.cat([x, c], dim=1))   # 3 + 5 = 8 个通道
```

G 是"编码器（2 次下采样到 32×32）→ 6 个残差块 → 解码器（2 次转置卷积回到 128×128）→ tanh"，共 843 万参数；D 是 6 层步长为 2 的卷积（128 → 2×2，2048 通道），共 4476 万参数，大头在最后那层卷积。

#### 代码 2|tools/train_stargan.py：训练一步 {#t11-c2}

```python
c_trg = c_org[torch.randperm(c_org.size(0))]        # 随机目标 = 打乱后的真实标签
# ---- D: WGAN-GP + 认属性 ----
out_src, out_cls = D(x_real)
d_loss_real, d_loss_cls = -out_src.mean(), classification_loss(out_cls, c_org)
with torch.no_grad():
    x_fake = G(x_real, c_trg)                        # 这一步不更新 G
d_loss_fake = D(x_fake)[0].mean()
alpha = torch.rand(x_real.size(0), 1, 1, 1, device=device)
x_hat = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
d_loss_gp = gradient_penalty(D(x_hat)[0], x_hat)
d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
# ---- G: 每 n_critic=5 步一次 ----
x_fake = G(x_real, c_trg)
out_src, out_cls = D(x_fake)
x_rec = G(x_fake, c_org)                             # 改回原属性
g_loss = -out_src.mean() + lambda_rec * (x_real - x_rec).abs().mean() \
         + lambda_cls * classification_loss(out_cls, c_trg)
```

梯度惩罚要对 D 的输入求导，而且这个导数本身还要再对 D 的参数求导（二阶导），所以 `torch.autograd.grad` 要加 `create_graph=True`。数据直接从 parquet 读：先打乱 row group 的顺序，再在 16 个 row group 的缓冲区内打乱，内存占用小。

本项目只做了一次"健全性检查"式的短训练：从随机初始化开始，batch 8，600 步，Intel Arc 核显（和其它任务共用，约 2 秒一步）。官方模型是 batch 16 训练 20 万步（约 20 个 epoch），我们的结果全部使用官方权重。

<figure class="fig"><img class="shot" src="img/gan_train_loss.jpg" alt="StarGAN 从头训练 600 步的损失曲线"><figcaption><b>从头训练 600 步的损失（浅色为每 10 步的平均，深色为滑动平均）。</b>梯度惩罚从 0.66 在 180 步内降到约 0.06，再缓慢回升到 0.15～0.18；Wasserstein 估计先回落到约 3.2，之后随 D 变强升到约 9；D 的分类损失降到 3.0 左右，低于随机猜测的 5 ln 2 ≈ 3.47（虚线）；循环重建 L1 从 0.49 稳步降到约 0.33。</figcaption></figure>

<figure class="fig"><img class="shot" src="img/gan_train_sample600.jpg" alt="训练 600 步时固定样本的生成结果"><figcaption><b>600 步时的样图：只学会了"模糊地复制输入"。</b>这时只看过 4800 张图（训练集的 3%），G 先学会了最容易的重建，属性几乎还改不动，还能看到转置卷积特有的棋盘格纹理。（这 8 张恰好是同一个人：parquet 的行按身份聚在一起，脚本现在改成每隔 50 行取一张。）</figcaption></figure>

结论：判断 GAN 训练是否正常，不能看某条损失有没有"降到 0"——对抗损失本来就不会收敛到 0。要看的是：梯度惩罚保持在小值、Wasserstein 估计不爆炸、分类损失低于随机水平、重建损失持续下降，同时定期看样图。

易错点：在 torch-xpu 2.15 上，用设备端 `torch.randperm(..., device="xpu")` 做索引会让核显报 DEVICE_LOST。改成在 CPU 上打乱再搬到设备后就正常了（官方代码本来也是这么做的）。

#### 代码 3|use_instance_stats：一个能让重建误差翻倍的陷阱 {#t11-c3}

推理时习惯写 `G.eval()`。但 StarGAN 的 InstanceNorm 打开了 `track_running_stats`：train 模式下用"这张图自己的均值方差"，eval 模式下改用训练中累计的平均值。官方训练和测试都一直处于 train 模式，那些平均值从来没参与过计算。

```python
def use_instance_stats(model):
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            m.track_running_stats = False     # eval() 时也用单张图自己的统计量
            m.running_mean = m.running_var = None
    return model
```

实测（同样 200 张 test 图，用原标签做重建）：用单图统计量时 L1 = 0.095，直接 `G.eval()` 时 L1 = 0.229，误差是 2.4 倍，图像出现明显的偏色、发糊和棋盘格。

<figure class="fig"><img class="shot" src="img/gan_instnorm.jpg" alt="InstanceNorm 两种统计量下的重建结果对比"><figcaption><b>同一个网络、同一份权重，只差一个 eval()。</b>上：输入；中：单图统计量（与训练时一致）；下：改用 running stats 后的输出。</figcaption></figure>

易错点：加载别人的权重时，前向方式必须和训练时一致。带 BatchNorm / InstanceNorm 的网络尤其要查 `track_running_stats` 和 train / eval 模式。一个快速的自检：用原标签做重建，误差应该很小。

#### 代码 4|edit_photo：贴回原图 {#t11-c4}

```python
face, A = warp_to_crop(img_bgr, kps)                 # A: 原图 -> 128 输入 (2x3 仿射)
outs = self.edit_many(face, edits)                   # 一次 batch 做多个编辑
inv = dict(dsize=(w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
mask = cv2.warpAffine(ellipse_mask(), A, **inv)[..., None]   # 羽化椭圆, 逆变换回原图
delta = o.astype(np.float32) - face.astype(np.float32)       # G 改了什么
res = img_bgr.astype(np.float32) + mask * cv2.warpAffine(delta, A, **inv)
```

`WARP_INVERSE_MAP` 表示"A 是原图到小图的矩阵，请按它的逆把小图铺回原图"，不用自己求逆矩阵。人脸很大时（缩放比 < 0.75），对齐前先用 `INTER_AREA` 把原图缩小，避免双线性缩小产生锯齿。编辑可以写预设名（`"blond_hair"`、`"flip_male"`、`"flip_young"`），也可以写字典（`{"Blond_Hair": 1, "Male": "flip"}`）。

#### 代码 5|tools/eval_stargan.py：自己算 FID {#t11-c5}

pytorch-fid 0.3.0 自带的 `calculate_frechet_distance` 调用了 `scipy.linalg.sqrtm(..., disp=False)`，而 scipy 1.18 删除了 `disp` 参数，一调用就报错。我们只用它的 Inception 网络提特征，距离自己算：

```python
def frechet_distance(mu1, s1, mu2, s2):
    w, v = np.linalg.eigh(s1)
    r = (v * np.sqrt(np.clip(w, 0, None))) @ v.T          # r = s1 的平方根 (对称)
    eig = np.linalg.eigvalsh(r @ s2 @ r)                   # 与 s1·s2 的特征值相同
    tr_covmean = np.sqrt(np.clip(eig, 0, None)).sum()      # = Tr((s1·s2)^(1/2))
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)
```

验算：在随机数据上与 `scipy.linalg.sqrtm` 的结果相差 3×10⁻¹¹。s1·s2 不是对称矩阵，直接开方数值上容易出复数；换成对称矩阵 r·s2·r 再求特征值就稳定了。

### 例题 {#t11-s3}

#### 例题 1|FID 手算 {#t11-e1}

**真实特征服从二维高斯 N((0,0), diag(1,4))，生成特征服从 N((3,4), diag(4,1))，求 FID。**

均值差的平方：3² + 4² = 25。两个协方差的迹都是 5。两个对角矩阵相乘还是对角矩阵 diag(4, 4)，开方得 diag(2, 2)，迹为 4。

```latex
\mathrm{FID}=25+(5+5-2\times 4)=\boxed{27}
```

结论：FID 由"均值差"和"形状差"两部分组成。即使均值完全相同，两个分布方向不同的椭圆也有距离（这里形状差贡献了 2）。一维时 N(0,1) 与 N(1,4) 的 FID = 1 + 1 + 4 − 2×2 = 2，`frechet_distance` 的计算结果也是 2.0。

#### 例题 2|刚开始训练时，损失应该是多少 {#t11-e2}

**网络刚初始化时，D 的属性 logits 都约等于 0，梯度也几乎为 0。估计 D/loss_cls 和 λ_gp·D/loss_gp。**

logit 为 0 时 sigmoid 输出 0.5，不论标签是 0 还是 1，每个属性的交叉熵都是 ln 2；5 个属性求和：

```latex
\mathcal{L}_{\text{cls}}=5\ln 2\approx\boxed{3.466},\qquad
\lambda_{\text{gp}}\,\mathcal{L}_{\text{gp}}=10\times(0-1)^2=\boxed{10}
```

验算：用刚初始化的网络（batch 16）试跑，前 5 步的平均值 D/loss_cls = 3.462、D/loss_gp = 0.996，与估计吻合。之后梯度惩罚很快降到 0.1 以下，说明 D 的梯度范数被拉到了 1 附近；分类损失降到 5 ln 2 以下，说明 D 开始"认属性"。

易错点：刚开始的损失明显偏离这些值（比如分类损失是 30），通常是标签格式错了（如用了 −1/1 而不是 0/1），或图像没归一化到 [−1, 1]。

#### 例题 3|构造目标标签 {#t11-e3}

**一位棕发年轻女性，c' = (黑, 金, 棕, 男, 年轻) = (0, 0, 1, 0, 1)。求"变金发并翻转性别"和"变黑发并变老"的目标标签。**

发色三选一：设金发为 1 时，黑发、棕发自动置 0；性别取反 0 → 1；没提到的"年轻"保持原样。

```latex
c_1=\boxed{(0,1,0,1,1)},\qquad c_2=\boxed{(1,0,0,0,0)}
```

`make_target` 的运行结果相同。易错点：没提到的属性一定要填原值。如果把她的"年轻"误填成 0，G 会在改发色的同时把她变老——StarGAN 会"执行"标签里的每一维。所以任意照片要先用 D 的分类头估计原属性。

#### 例题 4|坐标换算：对齐图上的左眼在网络输入的哪里 {#t11-e4}

**CelebA 对齐图（178×218）中左眼平均位置 (69.58, 111.59)。经过 CenterCrop(178) 和 Resize(128) 后，它在 128×128 输入中的坐标是多少？**

裁剪去掉上方 (218 − 178) / 2 = 20 行，y 减 20；再缩放 r = 128/178 ≈ 0.7191。按像素中心对齐（PIL 的约定），x' = (x + 0.5)·r − 0.5：

```latex
\begin{aligned}
x'&=(69.58+0.5)\times 0.7191-0.5=49.89\\
y'&=(111.59-20+0.5)\times 0.7191-0.5=65.72
\end{aligned}
\qquad\Rightarrow\qquad\boxed{(49.89,\;65.72)}
```

`crop128_matrix` 给出的也是 (49.89, 65.72)。延伸：一张照片里两眼相距 50 像素，而标准位置两眼相距 108.26 − 69.58 = 38.68 像素，所以"原图 → 128 输入"的缩放比为 38.68/50 × 0.7191 = 0.556，贴回时要放大约 1.8 倍。这正是 residual 贴回更清晰的原因：只有"变化量"被放大，原图细节不动。
