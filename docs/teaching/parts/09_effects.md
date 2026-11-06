## 专题九|实时特效：贴纸、美颜、美妆与卡通头像|478 点网格、相似/透视变换、导向滤波、局部变形、One Euro {#t9|九 实时特效}

> 特效模块把"人脸在哪、长什么样、做什么表情"变成画面上的效果：AR 贴纸跟着头转、磨皮美白瘦脸大眼、口红腮红眼线、再用表情系数驱动一个卡通头像。所有特效都建立在 MediaPipe 的 478 点 3D 网格上，每帧只检测一次，纯 CPU 上 640×480 开全套 7 个美颜美妆特效约 14 ms/帧（70 FPS）。代码在 `facevision/mesh.py` 和 `facevision/effects/`，演示脚本 `tools/realtime_demo.py`。

<nav class="map" aria-label="本专题目录"><dl>
<dt>知识要点</dt><dd><a href="#t9-k1">1 478 点网格</a><a href="#t9-k2">2 脸坐标系</a><a href="#t9-k3">3 贴纸变换</a><a href="#t9-k4">4 Alpha 合成</a><a href="#t9-k5">5 皮肤掩码</a><a href="#t9-k6">6 保边磨皮</a><a href="#t9-k7">7 美白曲线</a><a href="#t9-k8">8 瘦脸大眼</a><a href="#t9-k9">9 口红腮红</a><a href="#t9-k10">10 One Euro</a><a href="#t9-k11">11 卡通头像</a><a href="#t9-k12">12 顺序与性能</a></dd>
<dt>代码精读</dt><dd><a href="#t9-c1">mesh.py</a><a href="#t9-c2">stickers.py</a><a href="#t9-c3">beauty.py</a><a href="#t9-c4">makeup.py</a><a href="#t9-c5">filters + pipeline</a></dd>
<dt>例题</dt><dd><a href="#t9-e1">相似变换与合成</a><a href="#t9-e2">导向滤波</a><a href="#t9-e3">局部平移</a><a href="#t9-e4">One Euro</a></dd>
</dl></nav>

### 知识要点 {#t9-s1}

**1. 478 点网格、表情系数与头部姿态。** 专题四的 106 点只描了五官轮廓，做美颜美妆不够密。MediaPipe Face Landmarker 一次给三样东西：① 478 个 3D 点（468 个网格点 + 每只眼 5 个虹膜点），x、y 是归一化的图像坐标，z 是相对深度（越小越靠近相机，尺度和 x 相同，所以 z 也乘图像宽度换成像素）；② 52 个 blendshape 表情系数（苹果 ARKit 的定义，每个 0～1，例如 jawOpen 张嘴、eyeBlinkLeft 左眼闭合、mouthSmileLeft 左嘴角上扬）；③ 一个 4×4 面部变换矩阵，把"标准 3D 人脸模型"摆到相机坐标系里，左上 3×3 是旋转 R。VIDEO 模式要求时间戳递增，它会用上一帧的点直接裁出人脸区域，跳过检测，所以比逐帧独立的 IMAGE 模式更快更稳。从 R 读姿态：R 的第 3 列 f 是"脸的正前方"，第 1 列 a 是"左眼到右眼"的方向（MediaPipe 相机坐标 x 右、y 上、z 指向观众）： {#t9-k1}

```latex
\text{yaw}=\operatorname{atan2}(f_x,\,f_z),\qquad
\text{pitch}=\arcsin(f_y),\qquad
\text{roll}=\operatorname{atan2}(-a_y,\,a_x)
```

<figure class="fig"><img class="shot" src="img/t9_mesh_groups.jpg" alt="478 点网格与部位分组"><figcaption><b>478 点网格与部位分组。</b>左：全部三角面片；右：程序里用到的部位（脸轮廓、内外唇、眼睛与虹膜、眉毛、鼻子、脸颊、额头）和常用单点（红点）。每组下标都这样画出来逐一核对过。</figcaption></figure>

易错点：MediaPipe 官方的 LEFT_EYE 指"被拍者自己的左眼"，在画面里是右边。本项目统一用"图像中的左/右"（和 106 点的 LMK106 一致），所以 `MESH["left_eye"]` 是画面左边那只眼（下标 33、133……），对应的表情系数却是 eyeBlink**Right**。

**2. 以脸为尺子：脸坐标系。** 贴纸、腮红都要"跟着脸走"：脸变大贴纸变大，歪头贴纸跟着歪。做法是给每张脸建一个局部坐标系：原点 O 取两眼中心的中点，三根轴 X（水平）、Y（向下）、Z（向脑后）的方向取自姿态矩阵 R（换到图像坐标要把 y、z 取反），长度都等于两眼距离 IOD。脸上任意一点都写成 {#t9-k2}

```latex
P = O + u\,X + v\,Y + w\,Z
```

(u, v, w) 叫这个点的"脸坐标"，不随平移、缩放、转头而变。例如墨镜中心定在 (0, 0.05, −0.25)：两眼中点略低一点、往镜头方向凸出 0.25 个 IOD；礼帽定在额头顶点上方。

易错点：第一版用"额头顶到下巴"的向量当 Y 轴，结果一张嘴下巴往下动，帽子就跟着上下跳。R 是 MediaPipe 用全部网格点拟合出的刚体旋转，不受表情影响，稳定得多。

**3. 贴纸锚定：相似变换与透视变换。** 贴纸是一张 RGBA 图片，要把它的 4 个角映射到画面上。先在脸坐标里规定贴纸放在哪、多宽，再用上面的公式把 4 个角变到图像上，得到 4 个目标点，最后求"贴纸 4 角 → 目标 4 点"的变换矩阵。两种模式： {#t9-k3}

```latex
\text{2D 相似变换 (4 个自由度, 2 对点确定)}:\quad
\begin{bmatrix}x'\\ y'\end{bmatrix}=
s\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}
\begin{bmatrix}x\\ y\end{bmatrix}+\begin{bmatrix}t_x\\ t_y\end{bmatrix}
```

2D 模式只用两眼连线：s = 图像上两眼距离 / 贴纸上对应两点距离，θ = 两眼连线倾角（就是 roll）。它能跟随平移、缩放、歪头，但人转头时贴纸不会变窄。3D 模式用带深度的脸坐标系：转头时 X 轴的图像投影变短（约乘 cos yaw），贴纸自然被压扁；再按针孔相机做"近大远小"修正（c 为画面中心，f 为焦距，近似取图像宽，Δz 为相对两眼中点的深度差）：

```latex
x' = c + (x-c)\,\frac{f}{f+\Delta z},\qquad
\begin{bmatrix}x'\\ y'\\ 1\end{bmatrix}\propto H\begin{bmatrix}x\\ y\\ 1\end{bmatrix}\quad(H\ \text{为}\ 3\times 3\ \text{单应矩阵, 8 个自由度, 4 对点确定})
```

三种变换的关系：相似（4 自由度，2 对点）⊂ 仿射（6 自由度，3 对点，平行线保持平行）⊂ 透视（8 自由度，4 对点，平行线可以相交）。OpenCV 对应 `getAffineTransform` / `getPerspectiveTransform` 和 `warpAffine` / `warpPerspective`。贴纸要缩得很小时先从图像金字塔里挑一层接近目标大小的再变换，否则会有锯齿和闪烁。

<figure class="fig"><img class="shot" src="img/t9_stickers_pose.jpg" alt="贴纸跟随旋转缩放与转头"><figcaption><b>贴纸跟随 roll、缩放和 yaw。</b>上两行：把同一张照片旋转 −25°～30°、缩放 0.6～1.0 倍，2D 和 3D 两种模式都贴得住。第三行：转头的真人照片，3D 模式下墨镜随 yaw 变窄并带一点梯形，左侧几乎全侧脸的人也贴在了脸上；多张脸各算各的变换。</figcaption></figure>

类比：贴纸像一张印在透明胶片上的面具。2D 模式是把胶片平贴在屏幕上转一转、缩一缩；3D 模式是把胶片真的"戴"在脸前面，头一转从侧面看它就变窄了。

**4. Alpha 合成与"预乘"。** 贴纸每个像素除了颜色 F 还有不透明度 α（0 全透明，1 不透明）。把它盖到背景 B 上的"over"运算是 {#t9-k4}

```latex
\text{out}=\alpha F+(1-\alpha)B
\qquad\Longrightarrow\qquad
\text{out}=F'+(1-\alpha)B,\quad F'=\alpha F\ \ (\text{预乘 alpha})
```

为什么要预乘？warp 时要做双线性插值。透明像素的 RGB 通常是黑色 (0,0,0)，直接插值，贴纸边缘的颜色会被"拉黑"，出现一圈黑边。先把颜色乘上 α 再插值，透明像素贡献的就是真正的 0，边缘干净。

**5. 皮肤掩码。** 磨皮美白只该作用在皮肤上。掩码 = 脸轮廓多边形（大半径高斯羽化，和头发、背景之间柔和过渡）乘以"1 − 五官掩码"（眼睛、眉毛、嘴唇多边形各向外扩 0.08 IOD，小半径羽化，保证五官内部严格为 0）： {#t9-k5}

```latex
M = \left(G_{0.12\,\text{IOD}} * \text{脸轮廓}\right)\cdot\left(1-G_{0.04\,\text{IOD}} * \text{五官}\right)\in[0,1]
```

易错点：第一版把整个掩码用同一个大半径模糊，眼睛中心的掩码值还有 0.28，磨皮强度 0.7 时眼睛也被抹了 20%。两种区域用两个不同的模糊半径才能"边缘柔和、五官清晰"同时做到（单元测试要求眼睛、嘴巴中心 < 0.05）。

**6. 保边磨皮：双边滤波与导向滤波。** 普通高斯模糊把所有边缘一起抹掉，脸会糊成一团。"保边滤波"只抹平小起伏（痘印、毛孔、细纹），保留大边缘（眉毛、鼻翼、嘴唇轮廓）。双边滤波给每个邻居的权重 = 空间高斯 × 颜色高斯，颜色差太大（跨过边缘）的邻居权重接近 0。导向滤波（He 等，2010）假设输出在每个小窗口 ω 里是引导图 I 的线性函数，自引导（I = 输入 p）时系数有闭式解： {#t9-k6}

```latex
w_{ij}=\exp\!\left(-\frac{\|i-j\|^2}{2\sigma_s^2}-\frac{|I_i-I_j|^2}{2\sigma_r^2}\right),\qquad
a_\omega=\frac{\operatorname{var}_\omega(p)}{\operatorname{var}_\omega(p)+\varepsilon},\quad
b_\omega=(1-a_\omega)\,\bar p_\omega,\quad
q=\bar a\,p+\bar b
```

平坦区域方差远小于 ε，a ≈ 0，输出 ≈ 窗口均值（被抹平）；边缘处方差远大于 ε，a ≈ 1，输出 ≈ 原图（被保留）。ε 就是"多大的起伏算瑕疵"的阈值，本项目 ε = 0.03²（像素归一化到 0～1），窗口半径 0.1 IOD。导向滤波只用盒式滤波（窗口均值），复杂度与半径无关；再在缩小一半的图上算 a、b（快速导向滤波），速度快 4 倍。最后按 strength × 皮肤掩码与原图混合。

易错点：OpenCV 自带的 `cv2.ximgproc.guidedFilter` 用三通道彩色引导时要对每个像素解 3×3 线性方程。640×480 画面里 207×181 的人脸区域上它要 2.3 ms，手写的逐通道自引导 1.3 ms，再改成半分辨率算 a、b 只要 0.37 ms，和全分辨率结果平均只差 0.17 个灰度级。人脸越大差距越明显：放大 3 倍的 LFW 照片上彩色引导版要几十毫秒。

**7. 美白：LAB 空间的亮度曲线。** LAB 把颜色拆成亮度 L 和两个色度 a（绿—红）、b（蓝—黄）。只抬 L 就能变亮而不偏色；在 RGB 里三个通道一起加，肤色会发灰。用对数曲线（β = 1 + 4 × strength）： {#t9-k7}

```latex
L'=255\cdot\frac{\ln\!\left(1+(\beta-1)\,L/255\right)}{\ln\beta}
```

它把 0 映射到 0、255 映射到 255，中间调被抬高。strength = 0.3（β = 2.2）时 L = 64、128、200 分别变成 85、152、215。256 个输入值的结果事先算好存成查找表（LUT），每个像素查表即可。再把 a、b 往中性值 128 收 4.5%，减少泛红泛黄，最后按皮肤掩码混合。

提醒：美白本质上是改变肤色，对深肤色的人尤其不自然。演示里强度默认只有 0.3，产品里应当默认关闭、由用户自己决定。

**8. 局部变形：瘦脸与大眼（Gustafsson 1993）。** 两种变形都用"反向映射"：对输出图的每个像素 x，算出它该从原图哪个位置 u 取颜色，再用 `cv2.remap` 插值。（正向映射"把原图像素搬到新位置"会留下没被搬到的空洞。）局部平移（瘦脸）：圆心 c、半径 r，把 c 处的内容往 m 拖；局部缩放（大眼）：眼睛中心 c、半径 r，强度 a： {#t9-k8}

```latex
\begin{aligned}
\text{平移}:\quad u&=x-s\left(\frac{r^2-\|x-c\|^2}{r^2-\|x-c\|^2+\|m-c\|^2}\right)^{2}(m-c), & \|x-c\|&<r\\
\text{缩放}:\quad u&=c+(x-c)\left(1-a\left(\frac{\|x-c\|}{r}-1\right)^{2}\right), & \|x-c\|&<r
\end{aligned}
```

圆外 u = x 不动，圆周上位移连续地降到 0，所以没有接缝。缩放公式在圆心处系数为 1 − a：从更靠近中心的地方取像素，等于放大 1/(1 − a) 倍。瘦脸在每侧下颌线上取 3 个点，都往"鼻尖与下巴的中点"拖，半径 0.45 IOD；大眼半径 1.1 倍眼宽，a = 0.4 × strength。所有变形先叠加到同一张坐标图上，最后只 remap 一次（只插值一次，画面更清楚），remap 的范围取所有圆的外接矩形。

易错点：|m − c| 远小于 r 时，平移公式会让整个圆几乎整体平移，然后在圆周附近不到 1 像素宽的一圈里突然降到 0，画面上出现"台阶"（见例题 3）。所以拖动距离固定取 |m − c| = r，让衰减平滑，再用系数 s（strength）把位移整体缩小；整体缩小位移场不会产生折叠。另一个坑：remap 的范围比圆小，圆边上会留下断层。

**9. 口红、腮红、眼线。** 口红如果直接涂成纯色，唇纹和高光全没了，像贴了一块塑料。在 LAB 里换色：亮度保留原来的起伏（L − 均值），只把整体亮度往口红色挪一半；色度直接换成口红色，只保留 30% 原来的起伏。掩码 = 外唇轮廓 − 内唇轮廓（张嘴时不会涂到牙齿），再轻微羽化。 {#t9-k9}

```latex
L'=L+0.5\,(L_{t}-\bar L),\qquad a'=a_{t}+0.3\,(a-\bar a),\qquad b'=b_{t}+0.3\,(b-\bar b)
```

腮红是脸颊中心的一个二维高斯"光斑"。矩阵 A 的两列是脸坐标系 X、Y 轴的图像投影乘腮红半径，所以光斑是一个随歪头旋转、随转头压扁的椭圆；再乘皮肤掩码，侧脸时不会涂到背景上：

```latex
w(p)=\exp\!\left(-\tfrac12\left\|A^{-1}(p-c)\right\|^2\right),\qquad
\text{out}=(1-\alpha)\,\text{原图}+\alpha\,\text{腮红色},\quad \alpha=0.45\,s\,w(p)\,M(p)
```

眼线是沿上眼睑 9 个点画一条抗锯齿粗线（0.022 IOD），在外眼角沿"眼睑方向 + 稍微向上"拉出 0.12 IOD 的眼尾。

<figure class="fig"><img class="shot" src="img/t9_beauty_closeup.jpg" alt="美颜美妆特写"><figcaption><b>美颜与美妆特写（LFW 照片放大 3 倍）。</b>从左到右：原图、磨皮、美白、瘦脸 + 大眼、口红 + 腮红 + 眼线、全部。磨皮后眉眼和嘴唇依然清晰，口红停在嘴唇上、没有涂到牙齿。</figcaption></figure>

**10. One Euro 滤波：视频去抖。** 关键点逐帧独立预测，人不动时相邻两帧也会差 1～2 像素，贴纸和口红就会"抖"。指数滑动平均（EMA）的系数 α 小则平滑但跟不上，α 大则跟得上但抖。One Euro 滤波（Casiez 等，CHI 2012）让 α 随速度自适应：静止时截止频率低、强平滑；快速运动时截止频率高、弱平滑。速度先除以 IOD，参数就和分辨率、脸大小无关： {#t9-k10}

```latex
\hat x_t=\alpha\,x_t+(1-\alpha)\,\hat x_{t-1},\qquad
\alpha=\frac{1}{1+\tau/\Delta t},\qquad
\tau=\frac{1}{2\pi f_c},\qquad
f_c=f_{\min}+\beta\,\frac{|\dot{\hat x}|}{\text{IOD}}
```

本项目 f_min = 1 Hz、β = 2。在演示视频上实测：关键点抖动（二阶差分 |x_{t+1} − 2x_t + x_{t−1}| 的均值）从 0.020 降到 0.013 IOD/帧，少了 36%；平滑后与原始点的平均偏差 0.045 IOD（约 3 像素）。旋转矩阵 R 逐元素滤波后不再严格正交，用 SVD 投影回最近的旋转矩阵 R = UVᵀ。

口诀：先令 β = 0，调 f_min 直到静止不抖；再加大 β，直到运动不拖。

**11. 表情驱动的卡通头像。** 3D 动画里角色的脸 = 中性脸 + Σ 系数 × 动作形变，只要"系数"对齐，就能把真人的表情迁移到任何角色上（苹果的 Animoji 就是这样做的）。本项目不用 3D 模型，直接用 OpenCV 画一个矢量卡通脸，把系数映射成绘制参数（Right 后缀对应画面左边）： {#t9-k11}

```latex
\begin{aligned}
\text{眼睛张开度}&=\operatorname{clip}\left(1-1.15\,\text{eyeBlink}+0.4\,\text{eyeWide},\;0,\;1.3\right)\\
\text{嘴唇张开}&=1.2\,\text{jawOpen}+0.4\,\overline{\text{mouthUpperUp}}+0.3\,\overline{\text{mouthLowerDown}}\\
\text{嘴角上扬}&=\text{mouthSmile}-\text{mouthFrown}
\end{aligned}
```

yaw、pitch 让五官整体平移（假 3D），roll 让整张图绕头部中心旋转。先画 2 倍大再缩小，等于免费的抗锯齿。

<figure class="fig"><img class="shot" src="img/t9_avatar.jpg" alt="表情驱动卡通头像"><figcaption><b>表情驱动卡通头像（演示视频截图）。</b>右侧面板左上角的数值条就是驱动它的 blendshape 系数。只看 jawOpen 会漏掉"咬着牙露齿笑"，所以嘴唇张开还加上了 mouthUpperUp 和 mouthLowerDown。</figcaption></figure>

**12. 特效顺序与实测性能。** 顺序是固定的：先做逐像素的颜色处理（磨皮、美白、口红、腮红、眼线，它们用关键点算掩码，必须在几何变形之前），再做几何变形（瘦脸、大眼，已经画好的妆跟着一起变形，不会错位），最后贴贴纸、画网格（盖在最上层），卡通头像拼在右边。皮肤掩码每张脸每帧只算一次，几个特效共用。下表在 Intel Core Ultra 5 225H 上纯 CPU 实测（`tools/benchmark_effects.py`，每种分辨率 90 帧取中位数；1280×720 是把 960×540 的演示视频放大得到的，人脸像素更多；测量时机器上还有别的后台任务，CPU 占用约 30%～50%）： {#t9-k12}

<div class="scroll"><table class="wide"><thead><tr><th>阶段 / 组合（中位数）</th><th>640×480</th><th>1280×720</th></tr></thead><tbody>
<tr><td>MediaPipe 478 点 + 表情 + 姿态</td><td>6.7 ms</td><td>8.2 ms</td></tr>
<tr><td>One Euro 平滑</td><td>0.3 ms</td><td>0.2 ms</td></tr>
<tr><td>墨镜 / 礼帽 / 猫耳 / 胡子</td><td>0.4 / 0.8 / 0.5 / 0.2 ms</td><td>0.9 / 2.6 / 1.4 / 0.5 ms</td></tr>
<tr><td>磨皮 / 美白（各含 0.7～2.2 ms 皮肤掩码）</td><td>2.0 / 2.2 ms</td><td>8.7 / 6.7 ms</td></tr>
<tr><td>瘦脸 / 大眼</td><td>0.8 / 0.2 ms</td><td>1.8 / 0.5 ms</td></tr>
<tr><td>口红 / 腮红 / 眼线</td><td>0.3 / 1.4 / 0.2 ms</td><td>0.6 / 3.2 / 0.4 ms</td></tr>
<tr><td>卡通头像（含绘制）</td><td>6.4 ms</td><td>7.4 ms</td></tr>
<tr><td><b>端到端</b>：只检测</td><td>7.0 ms（144 FPS）</td><td>9.8 ms（102 FPS）</td></tr>
<tr><td><b>端到端</b>：墨镜 + 礼帽</td><td>8.4 ms（118 FPS）</td><td>13.5 ms（74 FPS）</td></tr>
<tr><td><b>端到端</b>：美颜 4 项 + 美妆 3 项</td><td>14.3 ms（70 FPS）</td><td>31.6 ms（32 FPS）</td></tr>
<tr><td><b>端到端</b>：卡通头像</td><td>13.9 ms（72 FPS）</td><td>18.3 ms（55 FPS）</td></tr>
</tbody></table></div>

端到端只算"检测 + 特效"，不含读摄像头、显示和写视频。贴纸和口红只处理一小块 ROI，几乎不花时间；真正贵的是 MediaPipe 本身和要处理整张脸的磨皮、美白，它们的耗时和人脸像素数成正比，所以 720p 下全套美颜刚好卡在 30 FPS。

易错点：第一次测速时数字忽高忽低，同一个检测有时 7 ms、有时 24 ms。排查发现是 Windows 11 的节能策略：没有前台窗口的命令行程序跑大约 2 秒后会被当成"后台任务"，调度到能效核并降频，sleep 也恢复不了。开着窗口的实时演示不受影响，所以测速脚本先用 `SetProcessInformation` 只对自己这个进程关掉节流（进程退出即失效，不改系统设置），否则所有数字都会慢 3 倍左右。

<figure class="fig"><img class="shot" src="img/t9_effects_grid.jpg" alt="全部特效总览"><figcaption><b>全部特效总览。</b>每行一张 LFW 照片，每列一个特效：墨镜、礼帽、猫耳、胡子、磨皮、美白、瘦脸 + 大眼、口红、腮红 + 眼线、全套美颜、网格、卡通头像。最后一行有两张脸，每张脸各自贴。演示视频取自美国 NIOSH 的公有领域采访视频（来源见 <code>assets/demo/SOURCE.md</code>），处理结果在 <code>outputs/effects/</code>。</figcaption></figure>

### 代码精读 {#t9-s2}

#### 代码 1|facevision/mesh.py：一次调用拿到网格、表情和姿态 {#t9-c1}

封装的工作只有两件：把 BGR 转成 RGB 喂给 MediaPipe，把它返回的归一化坐标、表情列表、矩阵整理成 numpy。

```python
if self.mode == "video":
    timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # 必须严格递增
    res = self.landmarker.detect_for_video(mp_img, timestamp_ms)
else:
    res = self.landmarker.detect(mp_img)
for i, lms in enumerate(res.face_landmarks):
    # 归一化坐标 [0,1] -> 像素坐标; z 和 x 同尺度, 所以也乘图像宽度
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
    bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
    M = np.asarray(res.facial_transformation_matrixes[i], dtype=np.float32)
    yaw, pitch, roll = pose_from_matrix(M)
```

`pose_from_matrix` 就是知识要点 1 的三个公式：`f = R[:, 2]`、`a = R[:, 0]`，`yaw = atan2(f[0], f[2])`。验算：Colin Powell 的照片从矩阵算出 roll = 3.5°，用两眼连线直接算是 3.3°，对得上。

#### 代码 2|facevision/effects/stickers.py：4 个角 → 变换矩阵 → 合成 {#t9-c2}

```python
O, X, Y, Z = face_frame(face)                          # 脸坐标系, 三根轴长度都是 1 个 IOD
B = _base_point(face, sticker.spec.base, O)            # 锚点: 两眼中点 / 额头顶 / 人中
P = face_to_image(sticker.corners_uvw(), B, X, Y, Z)   # 贴纸 4 个角 -> 3D 像素坐标 (4, 3)
xy = P[:, :2]
if mode == "3d":
    dz = P[:, 2] - O[2]
    xy = c + (xy - c) * (f / (f + dz))[:, None]        # 近大远小
...
H = cv2.getPerspectiveTransform(src, dst_roi)          # 4 对点 -> 3x3 单应矩阵
warped = cv2.warpPerspective(src_img, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
img[y0:y1, x0:x1] = alpha_composite_premul(img[y0:y1, x0:x1], warped)
```

只 warp 到贴纸覆盖的那块 ROI（`dst_roi` 是减去 ROI 左上角后的坐标），一张贴纸 0.2～1 ms。贴纸素材本身也是代码画的（PIL 在 4 倍画布上画再缩小，`make_sticker_assets()`），存在 `assets/stickers/`。

#### 代码 3|facevision/effects/beauty.py：导向滤波与局部变形 {#t9-c3}

手写的自引导导向滤波，和公式一一对应：

```python
box = lambda x: cv2.boxFilter(x, -1, (2 * rs + 1, 2 * rs + 1))   # 窗口均值
mean = box(small)
var = box(small * small) - mean * mean
a = var / (var + eps)
b = (1.0 - a) * mean
mean_a, mean_b = box(a), box(b)
mean_a = cv2.resize(mean_a, (w, h))                    # 在半分辨率上算 a、b, 放大回来
mean_b = cv2.resize(mean_b, (w, h))
return mean_a * p + mean_b
```

局部平移直接改"坐标图"（存的是每个输出像素去原图哪里取色），最后统一 remap：

```python
d2 = dx * dx + dy * dy                       # |x - c|^2
ratio[inside] = ((r2 - d2[inside]) / (r2 - d2[inside] + mc2)) ** 2
map_x -= strength * ratio * (mx - cx)
map_y -= strength * ratio * (my - cy)
...
img[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
```

#### 代码 4|facevision/effects/makeup.py：LAB 口红 {#t9-c4}

```python
lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
target = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
w = m / m.sum()
mean = (lab * w[..., None]).sum((0, 1))                    # 嘴唇区域的加权平均 L, a, b
new[..., 0] = lab[..., 0] + 0.5 * (target[0] - mean[0])   # 亮度: 保留起伏, 整体挪一半
new[..., 1] = target[1] + 0.3 * (lab[..., 1] - mean[1])   # 色度: 换成口红色
new[..., 2] = target[2] + 0.3 * (lab[..., 2] - mean[2])
colored = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
img[y0:y1, x0:x1] = blend(crop, colored, strength * m)
```

易错点：OpenCV 的 8 位 LAB 把 L 缩放到 0～255、a 和 b 加了 128 的偏移，和教科书上 L∈[0,100]、a/b 可正可负的定义不一样。这里只在同一种表示里做加减，所以没问题；如果要和别的软件对色值，必须先换算。

#### 代码 5|filters.py 与 pipeline.py：One Euro 和特效顺序 {#t9-c5}

```python
dx = (x - self.x_prev) / dt
a_d = smoothing_factor(dt, self.d_cutoff)
dx_hat = a_d * dx + (1 - a_d) * self.dx_prev                # 速度本身也低通一下
cutoff = self.min_cutoff + self.beta * np.abs(dx_hat) / scale  # 越快截止频率越高
a = smoothing_factor(dt, cutoff)                             # alpha = 1 / (1 + tau / dt)
x_hat = a * x + (1 - a) * self.x_prev
```

```python
EFFECT_ORDER = ["smooth", "whiten", "lipstick", "blush", "eyeliner", "slim", "big_eyes",
                "sunglasses", "hat", "cat_ears", "mustache", "mesh", "avatar"]
pipe = EffectPipeline(["smooth", "lipstick", "sunglasses"], mode="video")
out = pipe.process(frame_bgr, timestamp_ms)     # 每帧只跑一次 MediaPipe
```

多张脸时 `FaceSmoother` 先按脸中心的距离把这一帧的脸和上一帧配对（超过 1.5 IOD 就当新脸），每张脸有自己的滤波器。命令行演示：`python tools/realtime_demo.py --camera 0`，数字键 1～0 开关各个特效，`a` 切换卡通头像，`c` 原图对比，`p` 切换贴纸 2D/3D 模式，`f` 开关 One Euro。

### 例题 {#t9-s3}

#### 例题 1|两点求相似变换，再做一次 alpha 合成 {#t9-e1}

**墨镜素材 680×230，两个镜片中心在 (190, 112) 和 (490, 112)。画面里两只眼睛中心在 (200, 240) 和 (280, 300)。用 2D 相似变换把镜片中心对准眼睛：求 s、θ、平移 t，以及素材左上角 (0, 0) 落在哪。再求：镜片上一个颜色 (20, 20, 30)、α = 0.9 的像素盖在背景 (200, 180, 160) 上的结果。**

两眼向量 (80, 60)，长度 100；镜片向量 (300, 0)，长度 300。所以 s = 100/300 = 1/3，θ = atan2(60, 80) − 0 = 36.87°，cos θ = 0.8，sin θ = 0.6：

```latex
A=s\begin{bmatrix}0.8&-0.6\\0.6&0.8\end{bmatrix}=\begin{bmatrix}0.2667&-0.2\\0.2&0.2667\end{bmatrix},\qquad
t=q_1-A\,p_1=\begin{bmatrix}200-28.27\\240-67.87\end{bmatrix}=\boxed{\begin{bmatrix}171.7\\172.1\end{bmatrix}}
```

左上角 (0, 0) 经过变换就是 t 本身，即 (171.7, 172.1)。验算：A·(490, 112) + t = (280, 300)，正好是第二只眼。这个人头歪了约 37°，墨镜也跟着转了 37°。

合成用预乘：F′ = 0.9 × (20, 20, 30) = (18, 18, 27)，

```latex
\text{out}=F'+(1-\alpha)B=(18,18,27)+0.1\times(200,180,160)=\boxed{(38,\,36,\,43)}
```

结论：α = 0.9 的深色镜片几乎完全盖住背景，只透出 10%。`alpha_composite_premul` 跑出来也是 (38, 36, 43)。

#### 例题 2|导向滤波：什么被抹平，什么被保留 {#t9-e2}

**磨皮用 ε = 0.03² = 0.0009（像素已归一化到 0～1）。一个窗口落在脸颊上，像素标准差 0.02（约 5 个灰度级的毛孔起伏）；另一个窗口跨在眉毛和皮肤的交界处，标准差 0.2。分别求 a，并说明输出更像原图还是更像均值。**

```latex
a_{\text{脸颊}}=\frac{0.02^2}{0.02^2+0.0009}=\frac{0.0004}{0.0013}=\boxed{0.31},\qquad
a_{\text{眉毛}}=\frac{0.2^2}{0.2^2+0.0009}=\frac{0.04}{0.0409}=\boxed{0.98}
```

脸颊窗口 q ≈ 0.31 p + 0.69 × 均值，起伏只剩 31%，毛孔被抹平；眉毛边缘 q ≈ 0.98 p，几乎原样保留。窗口半径 r = 0.1 IOD，演示视频里 IOD ≈ 66 像素，所以 r = 7，窗口 15×15。

结论：ε 是"瑕疵"和"结构"的分界线，标准差远小于 √ε = 0.03 的起伏被抹掉，远大于它的保留。想磨得更狠就加大 ε，但皱纹、法令纹这种中等对比度的结构也会开始变平，看起来像塑料。

#### 例题 3|局部平移：为什么拖动距离不能太小 {#t9-e3}

**r = 30 像素，把圆心的内容向右拖。(a) |m − c| = 30；(b) |m − c| = 3。分别求离圆心 d = 0、15、27、29.9 像素处的位移 ratio × |m − c|（取 s = 1）。**

(a) |m − c|² = 900：

```latex
d=0:\ \left(\tfrac{900}{1800}\right)^2\times 30=7.50,\quad
d=15:\ \left(\tfrac{675}{1575}\right)^2\times 30=5.51,\quad
d=27:\ \left(\tfrac{171}{1071}\right)^2\times 30=0.77,\quad
d=29.9:\ 0.001
```

(b) |m − c|² = 9：

```latex
d=0:\ \left(\tfrac{900}{909}\right)^2\times 3=2.94,\quad
d=15:\ 2.92,\quad
d=27:\ \left(\tfrac{171}{180}\right)^2\times 3=2.71,\quad
d=29.9:\ 0.48
```

(a) 的位移从 7.5 平滑地降到 0；(b) 在 d = 27 时还有 2.71 像素，到圆周的最后 3 像素里突然降到 0，相当于把整个圆刚性平移、边上硬切一刀。

```latex
\text{结论}:\ \boxed{\text{拖动距离应与半径同量级, 力度用系数 } s \text{ 整体缩放}}
```

结论：这正是第一版瘦脸在下颌线上出现"台阶"的原因。项目里取 |m − c| = r，s = 0.5 × strength，圆心最大位移 0.25 r s。

#### 例题 4|One Euro 的 α：静止与运动 {#t9-e4}

**视频 30 FPS（Δt = 1/30 s），f_min = 1 Hz，β = 2。(a) 人静止时 α 是多少？(b) 头以每秒 2 个 IOD 的速度移动时呢？**

(a) 静止时 f_c = f_min = 1 Hz：

```latex
\tau=\frac{1}{2\pi\times 1}=0.159\ \text{s},\qquad
\alpha=\frac{1}{1+0.159\times 30}=\frac{1}{5.77}=\boxed{0.173}
```

(b) f_c = 1 + 2 × 2 = 5 Hz：

```latex
\tau=\frac{1}{2\pi\times 5}=0.0318\ \text{s},\qquad
\alpha=\frac{1}{1+0.0318\times 30}=\frac{1}{1.955}=\boxed{0.512}
```

静止时每帧只采纳 17% 的新观测，抖动被压得很低；运动时采纳 51%，延迟只有静止时的约三分之一。用 `smoothing_factor(1/30, 1.0)` 和 `smoothing_factor(1/30, 5.0)` 算出来是 0.1732 和 0.5115。

类比：One Euro 像一个会看路况的司机。车停着时方向盘握得很稳（强平滑），路上一有急弯就立刻放开手跟着打（弱平滑）。
