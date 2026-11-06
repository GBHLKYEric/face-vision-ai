## 专题四|关键点与人脸对齐|5 点 / 68 点 / 106 点 / 478 点，热图与坐标回归，相似变换与 Umeyama 算法 {#t4|四 关键点·对齐}

> 关键点告诉我们眼睛、鼻子、嘴巴在哪里。识别只需要 5 个点来"把脸摆正"；美颜、贴纸、3D 需要更密的点来描出五官轮廓。本专题的核心是对齐：一个只有几行的线性代数算法，却决定了识别准确率的上限。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t4-k1">1 各种点数</a><a href="#t4-k2">2 热图 vs 坐标回归</a><a href="#t4-k3">3 为什么要对齐</a><a href="#t4-k4">4 相似变换</a><a href="#t4-k5">5 Umeyama 算法</a><a href="#t4-k6">6 NME 指标</a></dd>
<dt>代码精读</dt><dd><a href="#t4-c1">align.py</a><a href="#t4-c2">landmark.py</a></dd>
<dt>例题</dt><dd><a href="#t4-e1">两点求相似变换</a><a href="#t4-e2">对齐后为何要翻转测试</a></dd>
</dl></nav>

### 知识要点 {#t4-s1}

**1. 关键点的几种"规格"。** 5 点（双眼、鼻尖、两嘴角）：检测器顺带输出，够做对齐。68 点（300W 标准）：轮廓 17 + 眉毛 10 + 鼻子 9 + 眼睛 12 + 嘴 20，学术评测最常用。106 点（本项目 2d106det）：比 68 点多了眼眶、嘴唇内圈的细节，适合美妆。478 点（MediaPipe Face Mesh）：覆盖整张脸的三角网格，还带深度 z，适合贴纸、3D 特效。点越多越精细，但也越慢、越难标注。 {#t4-k1}

**2. 热图回归 vs 坐标回归。** 坐标回归：网络最后一层直接输出 2N 个数（每个点的 x, y），简单快速，本项目的 2d106det、1k3d68 都是这种，输出范围 [−1, 1] 再映射回像素。热图回归（HRNet 等）：每个点输出一张小图，图上越亮的地方越可能是这个点，取最亮的位置作为坐标。热图保留了空间结构、精度更高，但输出大、后处理慢，而且坐标被量化到热图格子上（需要亚像素修正）。手机端实时应用多用坐标回归，学术榜单多用热图。 {#t4-k2}

**3. 为什么一定要对齐。** 识别网络在训练时看到的每张脸，眼睛都在 (38, 52) 和 (74, 52) 附近、嘴角都在 (42, 92) 和 (71, 92) 附近。如果测试时给它一张歪着头、离得远的脸，网络提取的特征会被"位置变化"污染。对齐就是把每张脸都变换到这个标准模板上，让网络只需要关心"这是谁"，不用关心"脸在哪、歪没歪"。 {#t4-k3}

类比：考试统一用答题卡，阅卷机才能准确读答案。对齐就是把每张脸"填进标准答题卡"。

**4. 相似变换。** 只允许旋转、等比缩放、平移（不允许拉伸、错切），这样脸的形状不会变，只是摆放方式变了。它有 4 个自由度：缩放 s、角度 θ、平移 (tₓ, t_y)，写成矩阵： {#t4-k4}

```latex
\begin{pmatrix}x'\\y'\end{pmatrix}=
\underbrace{\begin{pmatrix}s\cos\theta & -s\sin\theta\\ s\sin\theta & s\cos\theta\end{pmatrix}}_{sR}
\begin{pmatrix}x\\y\end{pmatrix}+\begin{pmatrix}t_x\\t_y\end{pmatrix}
```

5 个点对应 10 个方程、4 个未知数，方程比未知数多，一般不可能每个点都精确对上（检测本身有误差），所以求"总体误差最小"的那个变换——最小二乘。

**5. Umeyama 算法：最小二乘相似变换的闭式解。** 目标是让变换后的检测点和模板点的距离平方和最小： {#t4-k5}

```latex
\min_{s,R,t}\ \frac1n\sum_{i=1}^{n}\bigl\|\,y_i-(sRx_i+t)\,\bigr\|^2
```

不用迭代，四步直接算出答案：

```latex
\begin{aligned}
&\text{(1) 去重心: } \tilde x_i=x_i-\bar x,\quad \tilde y_i=y_i-\bar y\\
&\text{(2) 协方差并做 SVD: } \Sigma=\tfrac1n\textstyle\sum_i \tilde y_i\tilde x_i^{\top}=UDV^{\top}\\
&\text{(3) 旋转: } R=U\,S\,V^{\top},\quad S=\mathrm{diag}\bigl(1,\ \operatorname{sign}(\det U\det V)\bigr)\\
&\text{(4) 缩放与平移: } s=\frac{\operatorname{tr}(DS)}{\tfrac1n\sum_i\|\tilde x_i\|^2},\qquad t=\bar y-sR\bar x
\end{aligned}
```

直观理解：去掉重心就消掉了平移；剩下的"旋转+缩放"由两组点的相关矩阵决定，SVD 把它拆成"旋转 × 拉伸 × 旋转"，取出纯旋转部分 UVᵀ 就是最优旋转。第 (3) 步里的 S 用来防止算出镜像翻转（行列式为 −1 的"假旋转"）。

易错点：判断镜像要用 det(U)·det(V)，不要用 det(Σ)。只有 2 个点时 Σ 的秩是 1、行列式恒为 0，用 det(Σ) 判断会漏掉镜像。这正是本项目开发时踩到并修复的 bug（见例题 1）。

**6. 关键点评测指标 NME。** 归一化平均误差：每个点的预测位置和真值的距离取平均，再除以一个"脸的尺寸"（300W 用两眼外角距离，AFLW2000 用框的 √(宽×高)），让大脸小脸可以公平比较。3% 的 NME 大约相当于 100 像素宽的脸上平均差 3 个像素。 {#t4-k6}

```latex
\mathrm{NME}=\frac{1}{N}\sum_{k=1}^{N}\frac{\|\hat p_k-p_k\|_2}{d}
```

### 代码精读 {#t4-s2}

#### 代码 1|facevision/align.py：手写 Umeyama {#t4-c1}

```python
def umeyama(src, dst, estimate_scale=True):
    n, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean        # ① 去重心
    cov = dst_c.T @ src_c / n                             # ② 协方差
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:          # ③ 防镜像
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    scale = (S * d).sum() / src_c.var(0).sum()            # ④ 缩放
    T = np.eye(dim + 1)
    T[:dim, :dim] = scale * R
    T[:dim, dim] = dst_mean - scale * R @ src_mean        #    平移
    return T
```

对齐本身两行：求出 2×3 矩阵 M，再让 OpenCV 按 M 把整张图"搬"到 112×112 的画布上（每个输出像素反算回原图位置，双线性插值取色）：

```python
M = umeyama(kps, ARCFACE_TEMPLATE)[:2]
aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
```

单元测试 `tests/test_core.py` 用一个已知的旋转 + 缩放 + 平移生成点，检查 Umeyama 能把参数原样解出来（误差 10⁻⁸），这是验证数学代码最可靠的办法：先造一个知道答案的题，再看程序能不能做对。

#### 代码 2|facevision/landmark.py：裁剪—推理—反变换 {#t4-c2}

关键点模型的输入是 192×192 的方块。以检测框中心为圆心、框边长的 1.5 倍为边长裁出一块（多留出下巴和额头），缩放到 192，推理得到 [−1, 1] 的坐标，再用逆矩阵变回原图：

```python
scale = 192 / (max(w, h) * 1.5)
M = crop_transform(center, 192, scale)          # 原图 -> 裁剪图
crop = cv2.warpAffine(img, M, (192, 192))
pred = session.run(...)[0][0].reshape(-1, 2)    # 106 个点, 范围 [-1, 1]
pred = (pred + 1) * 96                          # -> [0, 192]
pts = transform_points(pred, cv2.invertAffineTransform(M))   # -> 原图坐标
```

`guess_normalization` 函数顺带演示了一个工程细节：有些从 MXNet 转来的老模型把"减均值除方差"写进了网络第一层（叫 bn_data），这种模型输入要给原始 0～255 像素；新模型则需要手动归一化。读网络前几个节点的名字就能判断，给错了不会报错，只会结果变差。

### 例题 {#t4-s3}

#### 例题 1|只用两只眼睛求相似变换 {#t4-e1}

**检测到左眼 (100, 120)、右眼 (140, 100)，模板左眼 (38.29, 51.70)、右眼 (73.53, 51.50)。求缩放、旋转角，并说明开发中遇到的镜像 bug。**

两只眼睛的连线：原图向量 (40, −20)，长度 44.72，角度 −26.57°（右眼比左眼高，头向右歪）。模板向量 (35.24, −0.19)，长度 35.24，角度 −0.32°（几乎水平）。

```latex
\begin{aligned}
s&=\frac{35.24}{44.72}=\boxed{0.788}\\
\theta&=-0.32^\circ-(-26.57^\circ)=\boxed{+26.25^\circ}\ \text{（把歪的头转正）}
\end{aligned}
```

Umeyama 算出的矩阵第二行是 (0.3485, 0.7067, −67.95)：0.7067 = 0.788 × cos 26.25°，0.3485 = 0.788 × sin 26.25°，与手算一致。

提醒：修复前的版本用 det(Σ) 判断镜像，两个点时 Σ 的行列式为 0 判断失效，得到的是一个行列式为负的"镜像矩阵"，把脸左右翻转了。5 个点时 Σ 满秩，两种写法结果相同，所以 LFW 的数字不受影响——但这类只在边界情况出错的 bug 最危险，只有专门构造测试才能发现。

#### 例题 2|为什么对齐后还要"翻转测试" {#t4-e2}

**评测时把对齐后的脸和它的水平镜像各提一次特征再相加，为什么有用？代价是什么？**

人脸近似左右对称，镜像后仍是同一个人。两次特征相加相当于对"脸的左右半边"做了平均，抵消了光照从一侧打来、轻微侧脸等不对称因素带来的噪声。本项目 R50 模型加不加翻转，LFW 准确率分别见专题七。代价是识别时间翻倍。

结论：这类"测试时增强"（TTA）几乎是免费的精度，所以论文和比赛都会用；实时系统里则常常省掉以换取速度。
