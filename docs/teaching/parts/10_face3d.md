## 专题十|3D 人脸重建：从一张照片恢复三维网格|3DMM、弱透视投影、3DDFA-V2 参数回归、Z-buffer 渲染 {#t10|十 3D 重建}

> 前面的专题都在 2D 图像上工作。本专题从一张普通照片恢复出 38365 个顶点的 3D 人脸网格：一个 3.3M 参数的 MobileNet 回归 62 个 3DMM 参数，参数代回公式得到网格；再用纯 numpy 写一个渲染器，把网格贴回照片、画成深度图、摆正后从任意角度转着看。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t10-k1">1 为什么能恢复 3D</a><a href="#t10-k2">2 3DMM</a><a href="#t10-k3">3 弱透视投影</a><a href="#t10-k4">4 62 个参数</a><a href="#t10-k5">5 3DDFA-V2</a><a href="#t10-k6">6 姿态角</a><a href="#t10-k7">7 光栅化</a><a href="#t10-k8">8 法向与光照</a><a href="#t10-k9">9 纹理</a><a href="#t10-k10">10 实测结果</a><a href="#t10-k11">11 展望</a></dd>
<dt>代码精读</dt><dd><a href="#t10-c1">参数→网格</a><a href="#t10-c2">相机矩阵→姿态</a><a href="#t10-c3">向量化 Z-buffer</a><a href="#t10-c4">光照与任意视角</a><a href="#t10-c5">对拍原版</a></dd>
<dt>例题</dt><dd><a href="#t10-e1">ROI 裁剪</a><a href="#t10-e2">读姿态角</a><a href="#t10-e3">Z-buffer</a><a href="#t10-e4">Lambert 光照</a></dd>
</dl></nav>

### 知识要点 {#t10-s1}

**1. 为什么一张照片能恢复 3D。** 拍照是把 3D 世界压扁成 2D，深度信息丢了：理论上有无数种 3D 形状投影出来都是同一张照片，这叫"病态问题"。破局的办法是加先验知识：人脸长得都差不多，两只眼睛一个鼻子，鼻子总是凸出来的。只要在"所有合理的人脸"这个很小的空间里，找一张投影后和照片最像的，3D 形状就被确定下来了。这个"人脸空间"就是 3DMM。 {#t10-k1}

<figure class="fig"><img class="shot" src="img/face3d_overview.jpg" alt="3D 重建总览: 输入、68 点、网格叠加、深度图、姿态"><figcaption><b>本项目的完整输出</b>（<code>tools/reconstruct_3d.py</code>）。上：近正脸（yaw −18°）；下：大角度侧脸（yaw +50°，pitch +20°）。从左到右：输入、投影回图像的 68 个 3D 关键点、带光照的网格半透明叠加、深度图（越亮越近）、头部坐标轴（红 x、绿 y、蓝 z = 鼻子朝向）。侧脸那一行里，被挡住的半边下颌线也画出来了，这是 3D 关键点和 2D 关键点的本质区别。</figcaption></figure>

**2. 3DMM：平均脸 + 形状基 + 表情基。** 3DMM（3D Morphable Model，三维形变模型）把每张脸都表示成同一套网格：顶点数相同、编号含义相同（第 i 号顶点在所有人脸上都是同一个解剖位置，例如鼻尖）。把 N 个顶点的 xyz 拉成一个 3N 维向量，任何一张脸都写成 {#t10-k2}

```latex
\mathbf{S}=\bar{\mathbf{S}}+\mathbf{A}_{\text{shp}}\,\boldsymbol{\alpha}+\mathbf{A}_{\text{exp}}\,\boldsymbol{\beta},
\qquad \bar{\mathbf{S}}\in\mathbb{R}^{3N},\;
\mathbf{A}_{\text{shp}}\in\mathbb{R}^{3N\times 40},\;
\mathbf{A}_{\text{exp}}\in\mathbb{R}^{3N\times 10},\; N=38365
```

平均脸和形状基来自 Basel Face Model（BFM，几百个 3D 扫描），表情基来自 FaceWarehouse，3DDFA 用的是去掉脖子的版本（38365 个顶点、76073 个三角形）。基是怎么来的？对齐好的几百张扫描脸，每张都是 3N = 115095 维空间里的一个点；这些点并不是到处乱撒，而是挤在一个很"扁"的区域里。PCA（主成分分析）找出这团点变化最大的几个方向：第 1 个方向可能是"整张脸变大"，第 2 个是"变宽"……前 40 个方向就能描述绝大部分人与人之间的差异，α 就是沿这些方向走多远。

类比：捏脸游戏里的滑杆。平均脸是默认角色，每根滑杆是一个基向量，α、β 是滑杆位置。网络不用输出 115095 个坐标，只要输出 50 个滑杆位置。

易错点：3DMM 能"白送"68 个关键点，正是因为顶点一一对应：68 个关键点永远是固定编号的那 68 个顶点。代码里只取这 68 个顶点对应的 204 行做矩阵乘法，比算整个网格少算 564 倍。

**3. 弱透视投影。** 真实相机是透视投影：x = fX/Z，越远越小。但人脸前后起伏只有十来厘米，到相机的距离却有几十厘米到几米，可以近似认为所有点的深度都是同一个 Z₀，于是透视退化成"旋转 + 统一缩放 + 平移"： {#t10-k3}

```latex
x=\frac{fX}{Z}\approx\frac{f}{Z_0}X=s\,X\quad(\Delta Z\ll Z_0)
\qquad\Longrightarrow\qquad
\mathbf{V}=s\,\mathbf{R}\,\mathbf{S}+\mathbf{t}=\underbrace{[\,s\mathbf{R}\mid\mathbf{t}\,]}_{\mathbf{P}\in\mathbb{R}^{3\times 4}}\begin{bmatrix}\mathbf{S}\\ \mathbf{1}^{\top}\end{bmatrix}
```

V 的前两行就是图像上的 (x, y)，第三行是深度，只在渲染判断"谁挡住谁"时用。3DDFA 把 sR（3×3）和 t（3×1）拼成 3×4 的相机矩阵 P，正好 12 个数。

**4. 62 个参数。** 网络输出 12（相机矩阵 P）+ 40（形状 α）+ 10（表情 β）= 62 个数。这些数的量级天差地别：P 里的旋转元素约 0.0004（BFM 单位很小，要乘一个很小的 s 才变成像素），平移约 60，形状系数能到 10⁵。一个神经元很难同时学会输出相差 9 个数量级的数，所以训练时把每一维都标准化成均值 0、方差 1，推理时再还原： {#t10-k4}

```latex
\mathbf{p}=\hat{\mathbf{p}}\odot\boldsymbol{\sigma}+\boldsymbol{\mu},\qquad
\mathbf{p}=(\underbrace{p_1,\dots,p_{12}}_{\mathbf{P}=[s\mathbf{R}\mid\mathbf{t}]},\;\underbrace{\alpha_1,\dots,\alpha_{40}}_{\text{形状}},\;\underbrace{\beta_1,\dots,\beta_{10}}_{\text{表情}})
```

μ、σ 存在 `param_mean_std_62d_120x120.pkl` 里，是在训练集上统计出来的。

**5. 3DDFA-V2：一个小网络直接回归参数。** 流程只有四步：SCRFD 检测框 → 放大 1.58 倍、中心下移 0.14 倍框长的正方形 ROI（检测框只到眉毛，网格还有额头）→ 缩放到 120×120、归一化 (x − 127.5)/128 → MobileNetV1（深度可分离卷积，327 万参数）输出 62 个数 → 代回 3DMM 公式。3DDFA-V2（Guo 等，ECCV 2020）训练时的巧思包括：把"顶点距离损失 VDC"和"按参数重要性加权的参数损失 WPDC"交替优化；额外加一个直接回归 68 点的分支做正则（权重文件里的 `fc_lm`，推理时丢掉）；用 3D 旋转合成短视频来训练稳定性。我们只移植推理部分。 {#t10-k5}

易错点：3DDFA 训练时的输入是 OpenCV 读出的 BGR 顺序，所以这里**不能**像检测器那样 swapRB。颜色通道搞反，网络照样输出结果，只是变差，很难发现。这类问题只能靠和原版逐元素对拍来排除（见代码 5）。

**6. 从相机矩阵读出姿态角。** 网络输出的 sR 是 9 个自由的数，并不保证是"缩放 × 旋转"。拆分方法：前两行的模长都约等于 s，各自归一化后当作 R 的前两行，第三行用叉乘补出来（保证是右手系）： {#t10-k6}

```latex
s=\tfrac12\left(\lVert\mathbf{r}_1\rVert+\lVert\mathbf{r}_2\rVert\right),\quad
\hat{\mathbf{r}}_1=\frac{\mathbf{r}_1}{\lVert\mathbf{r}_1\rVert},\quad
\hat{\mathbf{r}}_2=\frac{\mathbf{r}_2}{\lVert\mathbf{r}_2\rVert},\quad
\hat{\mathbf{r}}_3=\hat{\mathbf{r}}_1\times\hat{\mathbf{r}}_2
```

再把 R 分解成三个基本旋转的乘积，就能读出三个欧拉角：

```latex
\mathbf{R}=\mathbf{R}_z(\text{roll})\,\mathbf{R}_y(-\text{yaw})\,\mathbf{R}_x(\text{pitch})
\;\Longrightarrow\;
\text{yaw}=\arcsin R_{20},\quad
\text{pitch}=\operatorname{atan2}(R_{21},R_{22}),\quad
\text{roll}=\operatorname{atan2}(R_{10},R_{00})
```

符号约定（已用图验证）：yaw > 0 鼻子转向画面左侧；pitch > 0 低头；roll > 0 头在画面里逆时针歪。把照片左右镜像，yaw 和 roll 会变号而 pitch 不变，这是测试里的一条检查。

易错点：|yaw| 接近 90° 时 cos(yaw) → 0，上面的除法失效，叫"万向节死锁"，代码里单独处理。另外 R 只是近似正交：例题 2 中前两行的点积是 0.0106，不是 0。

**7. 光栅化：画家算法与 Z-buffer。** 渲染就是把 76073 个三角形画到像素上，核心难题是"同一个像素被好几个三角形盖住时显示谁"。正交投影下，屏幕坐标直接取顶点的 (x, y)，深度取 z（越大越近）。 {#t10-k7}

- 画家算法：按三角形的平均深度从远到近排序，依次用 `cv2.fillConvexPoly` 涂上去，近的自然盖住远的。直观，但两个三角形互相穿插时会出错（例题 3），而且一个三角形只能一个颜色。
- Z-buffer：每个像素记住"目前最近的深度"，新来的点更近才覆盖。GPU 就是这么做的。判断像素 p 在不在三角形里、并插值深度和颜色，靠重心坐标：

```latex
\mathbf{p}=w_0\mathbf{v}_0+w_1\mathbf{v}_1+w_2\mathbf{v}_2,\quad
w_0=\frac{\operatorname{area}(\mathbf{p},\mathbf{v}_1,\mathbf{v}_2)}{\operatorname{area}(\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2)},\quad
2\operatorname{area}(\mathbf{a},\mathbf{b},\mathbf{c})=(b_x-a_x)(c_y-a_y)-(c_x-a_x)(b_y-a_y)
```

三个 w 都 ≥ 0 说明 p 在三角形内；深度 z = Σwᵢzᵢ，颜色同样按 w 加权平均（Gouraud 着色，看起来是光滑曲面）。有向面积的正负还顺便告诉我们三角形是正面还是背面朝向相机，背面直接跳过（背面剔除），省一半计算。

<figure class="fig"><img class="shot" src="img/face3d_painter_zbuffer.jpg" alt="画家算法与 Z-buffer 对比"><figcaption><b>画家算法 vs Z-buffer。</b>左一：画家算法（每个三角形一个颜色）；中间两张是鼻子附近放大 4 倍，左为画家算法的平面着色，右为 Z-buffer + 重心坐标插值的 Gouraud 着色，后者更光滑；右一：Z-buffer 顺便得到的深度图。每个三角形只有 1～2 个像素大时，两者整体几乎一样（测试里两者覆盖区域的 IoU > 0.95）。</figcaption></figure>

**8. 法向与 Lambert 光照。** 一个面朝向光源时亮，侧对光源时暗。三角形的法向用两条边的叉积求；顶点法向取周围三角形法向的平均（叉积长度是面积的 2 倍，直接相加就是按面积加权），再归一化： {#t10-k8}

```latex
\mathbf{n}_f=(\mathbf{v}_1-\mathbf{v}_0)\times(\mathbf{v}_2-\mathbf{v}_0),\qquad
\mathbf{n}_i=\frac{\sum_{f\ni i}\mathbf{n}_f}{\left\lVert\sum_{f\ni i}\mathbf{n}_f\right\rVert}
```

亮度 = 环境光 + Lambert 漫反射 + Blinn-Phong 高光，L 是指向光源的单位向量，V 是指向相机的单位向量：

```latex
I=k_a+k_d\max(0,\mathbf{n}\cdot\mathbf{L})+k_s\max(0,\mathbf{n}\cdot\mathbf{H})^{p},\qquad
\mathbf{H}=\frac{\mathbf{L}+\mathbf{V}}{\lVert\mathbf{L}+\mathbf{V}\rVert}
```

类比：手电筒斜着照桌面，光斑被拉长，同样的光摊到更大的面积上，单位面积就暗了，暗多少正好是 cos θ = n·L。这就是 Lambert 余弦定律。

易错点：叉积的方向取决于三个顶点的排列顺序。图像坐标 y 轴朝下，3D 软件 y 轴朝上，翻转 y 是一次镜像，会把顺时针变成逆时针，法向就整体朝里了，渲染出来一片漆黑。代码用 `orient_triangles` 自动统一：网格整体朝着相机，所以面积加权的法向之和必须指向 +z，否则把所有三角形反过来。

<figure class="fig"><img class="shot" src="img/face3d_shading.jpg" alt="法向、Lambert、Phong、纹理四步"><figcaption><b>着色四步。</b>1 顶点法向画成颜色（x→红，y 朝上→绿，z→蓝），2 只有 Lambert 项 max(0, n·L)，背光的下颌线全黑，3 加上环境光 0.35 和高光，4 照片采样的纹理乘以较弱的光照。光从左上前方打来。</figcaption></figure>

**9. 纹理：从照片取颜色，再"摆正"。** 每个顶点投影到照片上的 (x, y) 不是整数，用双线性插值取周围 4 个像素的加权平均作为顶点颜色。两个坑：被鼻子挡住的顶点会采到鼻子的颜色；侧对相机的那半边脸在照片里被压缩，采到的是拉长的条纹，甚至背景。办法是给每个顶点一个可信度（用 Z-buffer 判断是否可见，乘以法向朝相机的程度 n_z），不可信时借用左右对称位置顶点的颜色（把平均脸沿 x 镜像后找最近顶点，得到对称编号 m(i)）： {#t10-k9}

```latex
c_i'=(1-b_i)\,c_i+b_i\,c_{m(i)},\qquad b_i=(1-w_i)\,w_{m(i)},\qquad
w_i=\min\!\left(1,\;\frac{\max(0,n_{z,i})\cdot\text{vis}_i}{0.35}\right)
```

把公式里的 R 换成单位阵，就得到"摆正"的正脸网格；再乘上自己选的 yaw/pitch 旋转，就能从任意角度看这张脸。

<figure class="fig"><img class="shot" src="img/face3d_multiview.jpg" alt="多视角渲染"><figcaption><b>摆正后的多视角渲染</b>（<code>multiview.jpg</code>）。前三行是纹理网格在 pitch = −20°/0°/+20°、yaw = −60°…+60° 下的样子，最后一行是石膏像。照片里看得见的一侧（左边几列）纹理清晰；看不见的一侧靠对称补全，能看出接缝和拉伸，这是单张照片重建的固有局限。</figcaption></figure>

**10. 本项目实测。** 正确性：把我们的移植版和原版 3DDFA-V2 的 `TDDFA` 类放在一起，给同一个 SCRFD 框，在 32 张 LFW 人脸上逐元素比较：62 个参数、ROI、68 个关键点、全部 38365 个稠密顶点的最大绝对误差都是 **0**（逐位相同），姿态角误差 3.6×10⁻¹⁵ 度。精度：标准评测集 AFLW2000-3D 的官方下载站点无法连接、Hugging Face 上也没有镜像，本项目没有实测 NME；原作者报告 MobileNetV1 版在 AFLW2000-3D 上 68 点 2D NME 约 3.56%。速度（笔记本 CPU，同时有其它训练任务在跑，数字会波动）： {#t10-k10}

| 步骤 | 实测耗时 |
| --- | --- |
| 拟合一张脸（裁剪 + CNN + 稠密重建 + 姿态，PyTorch 1 线程） | 11～28 ms |
| 网格叠加渲染（250×250 原图） | 30～45 ms |
| 网格叠加渲染（640×640） | 约 200 ms |
| 任意视角纹理渲染 256×256 / 640×640 | 45～90 ms / 150～360 ms |
| 画家算法（640×640，Python 循环） | 约 270 ms |

易错点：PyTorch 默认用满所有 CPU 核。机器空闲时多线程更快，但机器繁忙时线程之间互相等待：同样的网络 4 线程要 175 ms，1 线程只要 27 ms。这么小的网络，CLI 默认只用 1 个线程。

<figure class="fig"><img class="shot" src="img/face3d_multiface.jpg" alt="多人脸与失败案例"><figcaption><b>多人脸与失败案例。</b>四张脸逐个拟合；右上角那张脸被画面边缘切掉一半，又是大角度侧脸，网格只剩一条"窄片"，姿态也不可信（yaw −42°、pitch +45°、roll +48°）。回归方法对"训练时没见过"的输入没有自我纠错能力，这时应该靠检测置信度或关键点可见性把它过滤掉。</figcaption></figure>

**11. 展望：可微渲染、NeRF 与 3D 高斯泼溅。** 本专题的渲染器是"不可微"的：Z-buffer 的"选最近"和"像素在不在三角形内"都是离散判断，梯度传不回去。三条更前沿的路线： {#t10-k11}

- 可微渲染：把光栅化做成可求导的（软化边界与深度比较，如 PyTorch3D、nvdiffrast），就能把渲染结果和照片逐像素比较，梯度一路传回 3DMM 参数、光照和纹理。这叫"分析-合成"（analysis by synthesis），可以在没有 3D 标注的照片上自监督训练，Deep3DFaceRecon、DECA 都是这个思路：

```latex
\min_{\boldsymbol{\alpha},\boldsymbol{\beta},\mathbf{P},\boldsymbol{\gamma}}\;
\sum_{\mathbf{x}\in\text{脸部像素}}\left\lVert I_{\text{photo}}(\mathbf{x})-\operatorname{Render}(\boldsymbol{\alpha},\boldsymbol{\beta},\mathbf{P},\boldsymbol{\gamma})(\mathbf{x})\right\rVert_1
+\lambda\sum_{k=1}^{68}\lVert\mathbf{l}_k-\hat{\mathbf{l}}_k\rVert^2
```

- NeRF：不用网格，用一个神经网络表示"空间中每一点的颜色 c 和密度 σ"，沿每条视线采样并做体渲染。照片级真实，但需要多视角照片或视频、每个人单独优化，渲染也慢：

```latex
C(\mathbf{r})=\sum_{i=1}^{M}T_i\left(1-e^{-\sigma_i\delta_i}\right)\mathbf{c}_i,\qquad
T_i=\exp\!\Big(-\sum_{j<i}\sigma_j\delta_j\Big)
```

- 3D 高斯泼溅（3DGS）：用几十万个带颜色、透明度的 3D 椭球表示场景，渲染时投影到屏幕、按深度排序、从前往后做透明度混合。这其实是"画家算法"的复兴（只是每个"三角形"换成了半透明的高斯斑点），在 GPU 上能实时渲染。GaussianAvatars 等工作把高斯点绑定到 3DMM 网格的三角形上，用 3DMM 的表情参数驱动它们，把本专题的参数化模型和新表示结合了起来：

```latex
C=\sum_{i=1}^{K}\mathbf{c}_i\,\alpha_i\prod_{j<i}\left(1-\alpha_j\right)
```

结论：3DMM + 网格是"强先验、少参数、一张照片就够"；NeRF / 3DGS 是"弱先验、高保真、要多视角"。实际系统常把两者结合：3DMM 负责可控的形状和表情，神经表示负责细节和真实感。

### 代码精读 {#t10-s2}

#### 代码 1|facevision/face3d/tddfa.py：62 个参数变成 3.8 万个顶点 {#t10-c1}

推理的前半段是普通的"裁剪 → 归一化 → 前向"，最后一步把标准化的输出还原：

```python
def predict_params(self, img, roi_box):
    crop = crop_img(img, roi_box)                       # 超出图像的部分补黑
    crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)  # 120x120
    x = (crop.astype(np.float32) - 127.5) / 128.0       # 注意: 保持 BGR, 不 swapRB
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    out = self.model(x)[0].numpy().astype(np.float32)
    return out * self.param_std + self.param_mean       # p = p_hat * sigma + mu
```

后半段就是 3DMM 公式本身。BFM 把顶点按 [x0, y0, z0, x1, y1, z1, …] 存成一列，所以算完要按列优先（order="F"）变回 (3, N)：

```python
def reconstruct(self, params, roi_box, dense=True, frontal=False):
    P, alpha_shp, alpha_exp = parse_params(params)      # 12 + 40 + 10
    if dense:
        u, w_shp, w_exp = self.u, self.w_shp, self.w_exp                  # (3N,1) (3N,40) (3N,10)
    else:
        u, w_shp, w_exp = self.u_base, self.w_shp_base, self.w_exp_base   # 只取 68 点的 204 行
    shape = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order="F")  # S
    sR, t = P[:, :3], P[:, 3:]
    if frontal:                                          # 去掉旋转 = 摆正的正脸
        sR = np.float32(P2sRt(P)[0]) * np.eye(3, dtype=np.float32)
    return similar_transform(sR @ shape + t, roi_box, self.size)   # V = sRS + t, 再映射回原图
```

`similar_transform` 做三件事：网络坐标系 y 朝上，先 y → 120 − y；训练数据来自 MATLAB（下标从 1 开始），x、z 各减 1；最后按 ROI 边长 / 120 缩放并平移回原图（例题 1）。

#### 代码 2|facevision/face3d/tddfa.py：从相机矩阵拆出姿态 {#t10-c2}

```python
def P2sRt(P):
    t = P[:, 3]
    r1, r2 = P[0, :3], P[1, :3]
    s = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    R = np.stack([r1, r2, np.cross(r1, r2)])            # 第三行用叉乘补
    return s, R, t

def matrix2angle(R):
    if R[2, 0] > 0.998:                                 # yaw = +90 度, 万向节死锁
        yaw, roll = np.pi / 2, 0.0
        pitch = roll + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        yaw, roll = -np.pi / 2, 0.0
        pitch = -roll + atan2(R[0, 1], R[0, 2])
    else:
        yaw = asin(R[2, 0])
        pitch = atan2(R[2, 1] / cos(yaw), R[2, 2] / cos(yaw))
        roll = atan2(R[1, 0] / cos(yaw), R[0, 0] / cos(yaw))
    return yaw, pitch, roll
```

画姿态坐标轴时，模型的第 k 根轴在图像上的方向就是 R 的第 k 列的前两个分量，y 分量取负（图像 y 朝下）：`d = (R[0, k], -R[1, k]) * length`。正脸时蓝色的 z 轴正对镜头，投影成一个点；转头越厉害，蓝线越长。

#### 代码 3|facevision/face3d/render.py：不用 for 循环的 Z-buffer {#t10-c3}

Python 里逐个三角形、逐个像素循环会慢到几十秒。向量化的思路是：先按包围盒大小把三角形分桶（1、2、4、8… 像素），同一桶里每个三角形都枚举 k×k 个候选像素，一次性用数组算出所有"(三角形, 像素)"对的重心坐标：

```python
px = x0[t, None] + ox[None]                             # (三角形数, k*k) 个候选像素
py = y0[t, None] + oy[None]
w0 = ((xb - px) * (yc - py) - (xc - px) * (yb - py)) / area[t, None]
w1 = ((xc - px) * (ya - py) - (xa - px) * (yc - py)) / area[t, None]
w2 = 1.0 - w0 - w1
inside = (px <= x1[t, None]) & (py <= y1[t, None]) & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
```

然后是 Z-buffer 本身，只有三行：每个像素取最大的 z，"赢家"的重心坐标用来插值属性：

```python
np.maximum.at(depth, pix, z)                            # 每个像素保留最近 (z 最大) 的深度
win = z >= depth[pix]                                   # 谁是那个最近的
image[pix[win]] = np.einsum("ij,ijc->ic", w[win], attrs[tri[tid[win]]])   # 属性 = sum w_i * attr_i
```

`np.maximum.at` 是"带重复下标的原地取最大"：同一个像素出现很多次时，普通的 `depth[pix] = np.maximum(...)` 只会保留其中一次的结果，`.at` 才会逐个累计。整个网格在 640×640 上一次渲染约 0.15～0.35 秒。

#### 代码 4|facevision/face3d/render.py：法向、光照与任意视角 {#t10-c4}

顶点法向 = 周围三角形法向之和。用 `np.bincount` 按顶点编号累加，比 Python 循环快上百倍：

```python
fn = np.cross(v1 - v0, v2 - v0)                         # 面法向, 长度 = 2 倍面积 (自带面积加权)
idx = tri.ravel()
vn = np.stack([np.bincount(idx, np.repeat(fn[:, c], 3), len(verts)) for c in range(3)], axis=1)
vn = vn / np.linalg.norm(vn, axis=1, keepdims=True)
```

任意视角 = 绕网格中心旋转 + 正交投影。缩放只由网格自身的大小决定、不随角度变化，所以连续角度渲染成的转台视频不会抖：

```python
v = v - (v.min(0) + v.max(0)) / 2                       # 以包围盒中心为旋转中心
radius = np.linalg.norm(v, axis=1).max()
v = v @ rotation_yaw_pitch(yaw_deg, pitch_deg).T        # 先 yaw 再 pitch
normals = vertex_normals(v, tri)                        # 旋转后重新算法向, 光源固定在相机侧
screen = v * (fill * size / 2 / radius)
screen[:, :2] += size / 2
```

#### 代码 5|对拍原版 3DDFA-V2，以及一键运行 {#t10-c5}

移植别人的模型，最可靠的验证是"同样的输入，逐元素比较输出"。原版仓库的人脸检测（FaceBoxes）和渲染器（Sim3DR）需要编译 Cython，但它的 `TDDFA` 类本身是纯 Python，可以直接导入：把原版目录加进 `sys.path`，传入同一个 SCRFD 框，比较 `recon_vers` 的输出和我们的结果。32 张 LFW 人脸上，62 个参数、ROI、68 个关键点、38365 个顶点的最大误差全部是 0。

对拍顺便发现了原版数据的一个小瑕疵：68 个关键点在 BFM 里的下标本应是同一个顶点的 (3v, 3v+1, 3v+2)，但第 44 号点的 z 下标来自相邻的另一个顶点（12087 而不是 12383），所以"稀疏关键点"和"稠密网格里对应的顶点"在第 44 号点上差约 0.06 像素。测试里专门写明了这一点，而不是把容差调大糊过去。

```bash
python tools/reconstruct_3d.py --input data/lfw/Colin_Powell/Colin_Powell_0001.jpg
# -> outputs/face3d/Colin_Powell_0001/: overview.jpg landmarks68.jpg mesh_overlay.jpg depth.png
#    pose.jpg mesh.obj multiview.jpg turntable.mp4 result.json
python -m pytest tests/test_face3d.py -q                # 8 个测试: 形状、正交性、镜像、渲染、OBJ 往返
```

`mesh.obj` 每行 `v x y z r g b` 一个带颜色的顶点、`f i j k` 一个三角形（下标从 1 开始），可以直接拖进 MeshLab、Blender 或 Windows 自带的 3D 查看器里转着看。

### 例题 {#t10-s3}

#### 例题 1|ROI 裁剪：检测框怎样变成网络输入 {#t10-e1}

**SCRFD 在 Colin_Powell_0001.jpg 上检测到框 (70.77, 63.67, 160.08, 185.44)。按 3DDFA-V2 的策略求送进网络的正方形 ROI；网络 120×120 坐标系里的点 (60, 50) 对应原图哪里？**

框宽 89.31、高 121.77。边长取宽高平均再放大 1.58 倍并取整；中心在框中心基础上往下移 0.14 倍边长：

```latex
\begin{aligned}
\text{old}&=\tfrac12(89.31+121.77)=105.54, & \text{size}&=\lfloor 1.58\times105.54\rfloor=\lfloor 166.75\rfloor=166\\
c_x&=160.08-\tfrac{89.31}{2}=115.425, & c_y&=185.44-\tfrac{121.77}{2}+0.14\times105.54=139.33
\end{aligned}
```

```latex
\text{ROI}=(c_x-83,\;c_y-83,\;c_x+83,\;c_y+83)=\boxed{(32.43,\,56.33,\,198.43,\,222.33)}
```

网络里的 1 个像素对应原图 166/120 = 1.3833 个像素。点 (60, 50) 先做 x − 1、y → 120 − y，再缩放平移：

```latex
x=(60-1)\times1.3833+32.43=114.04,\qquad y=(120-50)\times1.3833+56.33=\boxed{153.16}
```

验算：代码 `roi_from_bbox` 输出 [32.425, 56.3306, 198.425, 222.3306]，一致。

#### 例题 2|从旋转矩阵读出姿态角 {#t10-e2}

**对 Colin_Powell_0001.jpg，P2sRt 拆出的旋转矩阵（保留 4 位）如下，求 yaw、pitch、roll，并说明脸朝哪边转。**

```latex
\mathbf{R}=\begin{bmatrix}0.9497&0.0661&0.3062\\-0.0362&0.9970&-0.0682\\-0.3098&0.0537&0.9492\end{bmatrix}
```

直接套公式（下标从 0 开始）：

```latex
\begin{aligned}
\text{yaw}&=\arcsin(-0.3098)=-18.05^\circ\\
\text{pitch}&=\operatorname{atan2}(0.0537,\;0.9492)=3.24^\circ\\
\text{roll}&=\operatorname{atan2}(-0.0362,\;0.9497)=-2.18^\circ
\end{aligned}
\qquad\boxed{(\text{yaw},\text{pitch},\text{roll})\approx(-18.0^\circ,\;3.2^\circ,\;-2.2^\circ)}
```

yaw < 0：鼻子转向画面右侧约 18°，与照片一致；pitch、roll 都很小，基本是平视、头没歪。代码输出 (−18.048°, 3.236°, −2.185°)。

验算：第一行和第二行的点积 0.9497×(−0.0362) + 0.0661×0.9970 + 0.3062×(−0.0682) ≈ 0.0106，夹角 arccos(0.0106) = 89.39°，说明网络给出的"旋转"只是近似正交。

#### 例题 3|Z-buffer：两个互相穿插的三角形 {#t10-e3}

**三角形 A 的顶点 (x, y, z) 为 (0, 0, 1)、(4, 0, 2)、(0, 4, 3)；三角形 B 的 xy 相同，三个顶点 z 都是 1.5（z 越大越近）。像素 (1, 1) 和 (0, 0) 分别显示哪个三角形？画家算法会怎样？**

像素 (1, 1) 对 A 的重心坐标：

```latex
2\,\text{area}=4\times4-0\times0=16,\quad
w_0=\frac{(4-1)(4-1)-(0-1)(0-1)}{16}=0.5,\quad
w_1=\frac{(0-1)(0-1)-(0-1)(4-1)}{16}=0.25,\quad
w_2=0.25
```

```latex
z_A(1,1)=0.5\times1+0.25\times2+0.25\times3=\boxed{1.75}>1.5=z_B
```

所以 (1, 1) 显示 A。像素 (0, 0) 正好是 A 的第一个顶点，z_A = 1 < 1.5，显示 B。Z-buffer 逐像素比较，两个都对。

画家算法按平均深度排序：A 的平均深度 2 > B 的 1.5，A 被认为"更近"，最后画、整个盖住 B，像素 (0, 0) 就错了。代码 `rasterize` 的实测结果：(1, 1) 深度 1.75、(0, 0) 深度 1.5、(2, 1) 深度 2.0。

结论：画家算法按"整个三角形"排序，Z-buffer 按"每个像素"比较。人脸网格的三角形很小、几乎不穿插，两者结果接近；场景复杂时只有 Z-buffer 可靠。

#### 例题 4|Lambert 光照手算 {#t10-e4}

**光照方向 L = (−0.35, −0.45, 1)（图像坐标 y 朝下，所以是从左上前方打光），k_a = 0.35，k_d = 0.6，不计高光。求三个法向的亮度：(a) 正对相机 n = (0, 0, 1)；(b) 朝画面右侧 n = (0.6, 0, 0.8)；(c) 朝画面左侧 n = (−0.6, 0, 0.8)。**

先把 L 归一化：|L| = √(0.1225 + 0.2025 + 1) = 1.1511，L̂ = (−0.3041, −0.3909, 0.8687)。

```latex
\begin{aligned}
(a)\;\; \mathbf{n}\cdot\hat{\mathbf{L}}&=0.8687, & I&=0.35+0.6\times0.8687=0.871\\
(b)\;\; \mathbf{n}\cdot\hat{\mathbf{L}}&=0.6\times(-0.3041)+0.8\times0.8687=0.5126, & I&=0.35+0.6\times0.5126=0.658\\
(c)\;\; \mathbf{n}\cdot\hat{\mathbf{L}}&=0.6\times0.3041+0.8\times0.8687=0.8774, & I&=0.35+0.6\times0.8774=0.876
\end{aligned}
```

```latex
\boxed{I_a=0.871,\quad I_b=0.658,\quad I_c=0.876}
```

朝向光源一侧的 (c) 比正脸 (a) 还亮一点，背光的 (b) 明显暗，这就是图里左脸亮、右脸暗的原因。代码 `lambert_phong` 加上高光项（k_s = 0.15，p = 20）后，(a) 变成 0.947：正对相机又接近光源方向的额头会出现一块高光。

易错点：n·L 为负说明这个面背对光源，必须截断为 0（max），否则背光面会"吸光"变得比环境光还暗。
