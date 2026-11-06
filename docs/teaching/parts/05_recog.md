## 专题五|人脸识别：把脸变成 512 个数|特征向量、余弦相似度、从 Softmax 到 ArcFace、困难样本挖掘 {#t5|五 识别·损失}

> 识别网络把一张 112×112 的对齐人脸变成一个 512 维的向量（特征 / embedding）。同一个人的两张照片，向量方向几乎相同；不同人则方向差很远。网络结构用 ResNet 就够了，真正让人脸识别从 95% 走到 99.8% 的，是损失函数的设计——这是本专题的重点。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t5-k1">1 验证与识别</a><a href="#t5-k2">2 特征与余弦相似度</a><a href="#t5-k3">3 为什么归一化</a><a href="#t5-k4">4 Softmax 的问题</a><a href="#t5-k5">5 三种间隔</a><a href="#t5-k6">6 几何图解</a><a href="#t5-k7">7 困难样本挖掘</a><a href="#t5-k8">8 度量学习</a><a href="#t5-k9">9 骨干网络</a></dd>
<dt>代码精读</dt><dd><a href="#t5-c1">losses.py</a><a href="#t5-c2">backbones.py</a><a href="#t5-c3">recognizer.py</a></dd>
<dt>例题</dt><dd><a href="#t5-e1">余弦相似度</a><a href="#t5-e2">ArcFace logit</a><a href="#t5-e3">困难样本</a><a href="#t5-e4">为什么不用 Softmax 直接分类</a></dd>
<dt>实验</dt><dd><a href="#t5-r1">损失函数消融</a></dd>
</dl></nav>

### 知识要点 {#t5-s1}

**1. 验证（1:1）与识别（1:N）。** 验证回答"这两张是不是同一个人"（刷脸支付、手机解锁）：算一次相似度，和阈值比。识别回答"这个人是库里的谁"（门禁、相册聚类）：和库里每个人比，取最像的，最像的分数还要超过阈值，否则判为陌生人。两者底层都是"比较两个特征向量"，见 `facevision/gallery.py`。 {#t5-k1}

**2. 特征向量与余弦相似度。** 把 512 维特征想象成一个从原点出发的箭头。两个箭头的夹角越小越像，用夹角的余弦衡量： {#t5-k2}

```latex
\cos\theta=\frac{a\cdot b}{\|a\|\,\|b\|}=\frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\sqrt{\sum_i b_i^2}}\in[-1,1]
```

把每个向量先除以自己的长度（L2 归一化），点积就直接等于余弦。本项目在 LFW 上用 10 折交叉验证选出的阈值约为 0.25：余弦大于 0.25 判为同一人。

**3. 为什么要归一化特征。** 不归一化时，特征的"长度"会混进无关信息：清晰正脸的特征往往更长，模糊侧脸更短。比较时我们只想要"方向"（是谁），不要"长度"（多清晰）。归一化后所有特征都落在一个单位超球面上，比较只看夹角，训练和测试用的是同一把尺子。 {#t5-k3}

**4. 普通 Softmax 的问题。** 训练时把每个人当成一类（CelebA 有 10177 类），最后接一个全连接分类层，用交叉熵训练。第 j 类的打分（logit）是 Wⱼ·x + bⱼ。问题在于：Softmax 只要求"分对"，不要求"分得开"。训练集里每个人只要比别人高一点点就满足了，类内特征可以很松散；测试时遇到训练集里没有的新人，松散的特征就容易和别人混在一起。 {#t5-k4}

类比：Softmax 像考试 60 分及格就行；人脸识别需要的是"每科都远超及格线"，这样换一套没见过的卷子（新的人）才不会挂。

**5. 三种"间隔"：SphereFace、CosFace、ArcFace。** 先把特征 x 和每类权重 Wⱼ 都归一化，logit 就变成 cosθⱼ（θⱼ 是特征和第 j 类"中心方向"的夹角），再乘一个尺度 s（=64，否则数值太小 softmax 学不动）。训练时故意给正确类"使绊子"，让它的 logit 变小，逼网络把特征往类中心压得更紧： {#t5-k5}

```latex
\begin{aligned}
\text{NormFace:}\quad & s\cos\theta_y\\
\text{SphereFace:}\quad & s\cos(m_1\theta_y)\qquad\text{（角度乘 } m_1\text{，乘性角度间隔）}\\
\text{ArcFace:}\quad & s\cos(\theta_y+m_2)\qquad\text{（角度加 } m_2=0.5\text{，加性角度间隔）}\\
\text{CosFace:}\quad & s(\cos\theta_y-m_3)\qquad\text{（余弦减 } m_3=0.35\text{，加性余弦间隔）}
\end{aligned}
```

非目标类不变，仍是 s·cosθⱼ。之后照常做 softmax + 交叉熵：

```latex
L=-\log\frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)}+\sum_{j\neq y}e^{s\cos\theta_j}}
```

测试时不加间隔，于是每个人都有了 m 那么大的"安全余量"。ArcFace 的间隔直接加在角度上，几何意义最清晰（超球面上的测地距离），训练也最稳定，成为工业界默认选择。

**6. 几何图解。** 下图是二分类时特征在单位圆上的分布。Softmax 的决策边界是一条线，两类特征可以紧贴着边界；ArcFace 要求特征离边界至少 m 个弧度，两类之间留出一条"空白带"。 {#t5-k6}

<figure class="fig"><div class="scroll"><svg class="dg" style="width:100%;min-width:560px" viewBox="0 0 640 250" role="img" aria-label="Softmax 与 ArcFace 决策边界对比">
<circle class="ln" cx="150" cy="130" r="95"/><circle class="ln" cx="470" cy="130" r="95"/>
<line class="ln dash" x1="150" y1="130" x2="150" y2="25"/>
<line class="bm main" x1="150" y1="130" x2="217" y2="63"/><line class="bm main" x1="150" y1="130" x2="83" y2="63"/>
<text class="hl" x="222" y="58">W₁</text><text class="hl" x="60" y="58">W₂</text>
<circle class="bd" cx="160" cy="37" r="4"/><circle class="bd" cx="175" cy="40" r="4"/><circle class="bd" cx="195" cy="48" r="4"/>
<circle class="pl" cx="140" cy="37" r="4"/><circle class="pl" cx="125" cy="40" r="4"/><circle class="pl" cx="105" cy="48" r="4"/>
<text x="150" y="245" text-anchor="middle" class="nm">Softmax：两类可以贴着边界</text>
<line class="bm main" x1="470" y1="130" x2="537" y2="63"/><line class="bm main" x1="470" y1="130" x2="403" y2="63"/>
<line class="ln dash" x1="470" y1="130" x2="500" y2="38"/><line class="ln dash" x1="470" y1="130" x2="440" y2="38"/>
<text class="hl" x="542" y="58">W₁</text><text class="hl" x="380" y="58">W₂</text>
<circle class="bd" cx="520" cy="55" r="4"/><circle class="bd" cx="530" cy="62" r="4"/><circle class="bd" cx="512" cy="50" r="4"/>
<circle class="pl" cx="420" cy="55" r="4"/><circle class="pl" cx="410" cy="62" r="4"/><circle class="pl" cx="428" cy="50" r="4"/>
<text x="470" y="22" text-anchor="middle" class="qt">间隔 m</text>
<text x="470" y="245" text-anchor="middle" class="nm">ArcFace：中间留出 m 的空白带</text>
</svg></div><figcaption><b>间隔把同类压紧、异类推开。</b>紫色点属于第 1 类、灰色点属于第 2 类；虚线之间是 ArcFace 强制留出的角度间隔。</figcaption></figure>

**7. 困难样本挖掘。** 训练到后期，大多数样本已经分得很好，损失接近 0，梯度几乎全来自少数"难例"（侧脸、模糊、长得像的不同人）。挖掘的思路就是让难例多说话。本项目实现了 CurricularFace（课程学习式挖掘）：如果某个错误类 j 的 cosθⱼ 已经超过了正确类加间隔后的值，说明这是一个困难负类，把它的 logit 放大成 cosθⱼ·(t + cosθⱼ)；t 是训练进度（正确类 cosθ 的滑动平均），前期 t 小、难例权重低，先学简单的；后期 t 大，专攻难例——像人学习一样由易到难。 {#t5-k7}

```latex
N(t,\cos\theta_j)=\begin{cases}\cos\theta_j, & \cos(\theta_y+m)\ge\cos\theta_j\ \text{（容易）}\\ \cos\theta_j\,(t+\cos\theta_j), & \cos(\theta_y+m)<\cos\theta_j\ \text{（困难）}\end{cases}
```

另一种常见做法是在采样上挖掘：每批固定取 P 个人、每人 K 张（PK 采样），保证批内有同类可比，再在批内找"最远的同类"和"最近的异类"，见下一条。

**8. 度量学习：对比损失与三元组损失。** 不经过分类层，直接约束特征之间的距离。对比损失（DeepID2）：同一人距离越小越好，不同人距离至少拉开 margin。三元组损失（FaceNet）：锚点 a、同类 p、异类 n，要求 d(a,p) + margin < d(a,n)。三元组的数量随数据量爆炸式增长，随机挑大多是"太容易"的，所以必须配合批内困难样本挖掘（batch-hard）。分类式损失（ArcFace）一次和所有类中心比较，训练效率高得多，这是它取代三元组成为主流的原因。 {#t5-k8}

```latex
L_{\text{triplet}}=\max\bigl(0,\ \|f_a-f_p\|-\|f_a-f_n\|+\alpha\bigr)
```

**9. 骨干网络：IResNet 与 MobileFaceNet。** IResNet 是 ArcFace 论文为人脸改过的 ResNet：第一层用 3×3 卷积且不下采样（112 的小图经不起一上来就缩小 4 倍）；残差块是"BN-卷积-BN-PReLU-卷积-BN"；结尾不用全局平均池化，而是 BN → 全连接 → BN 直接得到 512 维（人脸各个位置的含义不同，平均池化会抹掉"眼睛在左上"这种空间信息）。MobileFaceNet 用深度可分离卷积把计算量降到约 1/8，结尾用 7×7 的"全局逐通道卷积"（GDConv）让网络自己学每个位置的权重。 {#t5-k9}

### 代码精读 {#t5-s2}

#### 代码 1|facevision/training/losses.py：一个类统一四种损失 {#t5-c1}

```python
def cosine(self, emb):
    # 特征和每类权重都归一化后做矩阵乘法, 一次得到对所有类的 cosθ
    return F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)

def margin_logits(self, cos, labels):
    target = cos.gather(1, labels[:, None])            # 取出正确类的 cosθ
    theta = torch.acos(target)
    new_target = torch.cos(theta + self.m2) - self.m3  # ArcFace(m2) / CosFace(m3)
    # θ+m 超过 π 后 cos 反而上升, 会"奖励"错得离谱的样本 -> 改用线性惩罚保证单调
    new_target = torch.where(target > math.cos(math.pi - self.m2), new_target,
                             target - self.m2 * math.sin(self.m2) - self.m3)
    logits = cos.scatter(1, labels[:, None], new_target)   # 只替换正确类那一列
    return logits * self.s
```

clamp 到 (−1, 1) 是为了 acos 不出 NaN（acos 在 ±1 处导数无穷大）。scatter 只改正确类那一列，其余类保持 cosθⱼ。单元测试检查了两件事：正确类的 logit 一定变小、其余类一定不变；SphereFace 的 ψ(θ) 在 [0, π] 上单调递减。

#### 代码 2|facevision/training/backbones.py：IResNet 残差块 {#t5-c2}

```python
def forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = self.bn1(x)
    out = self.prelu(self.bn2(self.conv1(out)))
    out = self.bn3(self.conv2(out))
    return out + identity          # 残差: 网络只学"变化量", 梯度可以直接流回浅层
```

初始化时把每个块最后一个 BN 的缩放系数设为 0：训练开始时每个残差块都是恒等映射，100 层的网络也像 1 层一样好训，然后再慢慢"长出"有用的变化。

#### 代码 3|facevision/recognizer.py：推理只有三步 {#t5-c3}

```python
blob = cv2.dnn.blobFromImages(faces, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)  # [-1,1], RGB
feats = session.run(...)[0]                       # (N, 512)
feats = feats + session.run(... flipped ...)[0]   # 翻转测试
return feats / np.linalg.norm(feats, axis=1, keepdims=True)
```

### 例题 {#t5-s3}

#### 例题 1|余弦相似度 {#t5-e1}

**a = (3, 4, 0)，b = (4, 3, 0)，求余弦相似度。**

```latex
\cos\theta=\frac{3\cdot4+4\cdot3+0}{5\times5}=\frac{24}{25}=\boxed{0.96}
```

两个向量长度都是 5，夹角约 16°，非常"像"。如果把 a 放大成 (30, 40, 0)，余弦不变，欧氏距离却变了很多——这就是比较人脸要用余弦（或归一化后再算距离）的原因。

#### 例题 2|ArcFace 到底让损失变大了多少 {#t5-e2}

**某样本对正确类 cosθ = 0.8，对另一个类 cosθ = 0.3，s = 64，只有这两类。分别用 Softmax（归一化版）、CosFace（m = 0.35）、ArcFace（m = 0.5 弧度）计算正确类的 logit、概率和损失。**

ArcFace：θ = arccos 0.8 = 36.87°，加上 0.5 弧度（28.65°）得 65.52°，cos 65.52° = 0.4144。

```latex
\begin{aligned}
\text{Softmax: }& 64\times0.8=51.2, & p&=\frac{1}{1+e^{-(51.2-19.2)}}\approx1, & L&\approx1.3\times10^{-14}\\
\text{CosFace: }& 64\times(0.8-0.35)=28.8, & p&=0.99993, & L&=6.8\times10^{-5}\\
\text{ArcFace: }& 64\times0.4144=26.52, & p&=0.99934, & L&=\boxed{6.6\times10^{-4}}
\end{aligned}
```

结论：对 Softmax 来说，这个样本已经"完美"了，梯度几乎为 0，不会再学；ArcFace 下它还有可观的损失，网络会继续把它往类中心推。间隔就是"永不满足"的老师。

#### 例题 3|困难样本 {#t5-e3}

**若正确类 cosθ 只有 0.5，另一类 cosθ = 0.45（长得很像的另一个人），重新计算 Softmax 与 ArcFace 的损失。**

```latex
\begin{aligned}
\text{Softmax: }& 64\times0.5=32.0\ \text{vs}\ 28.8, & L&=0.040\\
\text{ArcFace: }& 64\cos(60^\circ+28.65^\circ)=1.51\ \text{vs}\ 28.8, & L&=\boxed{27.3}
\end{aligned}
```

Softmax 认为这个样本"已经分对了，损失 0.04 不用管"；ArcFace 给出 27.3 的巨大损失，强迫网络把这两个人分开。困难样本自动获得了几百倍的关注度——ArcFace 本身就带有一种隐式的难例挖掘。

#### 例题 4|为什么不直接训练一个 1 万类的分类器来"识别" {#t5-e4}

**CelebA 有 10177 个人，训练一个 10177 类的分类网络，能不能直接用来认 LFW 里的人？**

不能。LFW 的 5749 人和 CelebA 的人完全不重叠，分类层的 10177 个输出没有一个对应 LFW 里的人。人脸识别是开放集问题：测试时会遇到训练时从未见过的人。所以训练完要把分类层扔掉，只留下 512 维特征提取器，靠"特征之间的距离"来判断——分类只是训练特征的手段，不是目的。

面试答法：人脸识别是开放集度量学习问题，分类损失（ArcFace）只是用来学到判别性特征；部署时去掉分类头，用特征余弦相似度做 1:1 验证或 1:N 检索。

### 消融实验：只换损失函数 {#t5-s4}

#### 实验|五种损失函数在同一条件下的对比 {#t5-r1}

{{ABLATION}}

{{ABLATION_NOTE}}

{{ABLATION_CAVEAT}}
