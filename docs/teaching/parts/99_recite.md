## 必背清单|必背清单|公式 · 答辩/面试标准答法 · 常见失分点 {#recite|必背清单}

> 把整个项目压缩成一页：公式背下来能推，问答背下来能讲，失分点背下来能避坑。

### 公式 {#r-s1}

<div class="scroll"><table class="wide"><thead><tr><th>名称</th><th>公式</th><th>一句话</th></tr></thead><tbody>
<tr><td>IoU</td><td>|A∩B| / (|A|+|B|−|A∩B|)</td><td>框的重叠程度，0.5 算框对</td></tr>
<tr><td>锚点解码</td><td>x₁ = cₓ − l·s，x₂ = cₓ + r·s</td><td>预测相对锚点的距离，乘步长</td></tr>
<tr><td>相似变换</td><td>y = sRx + t</td><td>4 个自由度：缩放、旋转、两个平移</td></tr>
<tr><td>Umeyama</td><td>Σ = UDVᵀ，R = U·diag(1, sign(det U det V))·Vᵀ</td><td>最小二乘对齐的闭式解</td></tr>
<tr><td>余弦相似度</td><td>a·b / (‖a‖‖b‖)</td><td>只看方向不看长度</td></tr>
<tr><td>ArcFace</td><td>s·cos(θ_y + m)，s = 64，m = 0.5</td><td>角度加间隔</td></tr>
<tr><td>CosFace</td><td>s·(cos θ_y − m)，m = 0.35</td><td>余弦减间隔</td></tr>
<tr><td>SphereFace</td><td>s·cos(m·θ_y)</td><td>角度乘间隔</td></tr>
<tr><td>三元组</td><td>max(0, d(a,p) − d(a,n) + α)</td><td>同类近、异类远</td></tr>
<tr><td>蒸馏</td><td>λ·(1 − cos(f_s, f_t))</td><td>学生特征对齐老师</td></tr>
<tr><td>BN</td><td>γ(x − μ)/√(σ²+ε) + β</td><td>推理时是固定线性变换</td></tr>
<tr><td>BN 融合</td><td>W′ = W·k，b′ = (b − μ)k + β，k = γ/√(σ²+ε)</td><td>合并进卷积，零精度损失</td></tr>
<tr><td>量化</td><td>q = round(x/scale) + zp，x ≈ (q − zp)·scale</td><td>FP32 → INT8，模型小 4 倍</td></tr>
<tr><td>TPR / FPR</td><td>TP/(TP+FN)，FP/(FP+TN)</td><td>通过率 / 误识率</td></tr>
<tr><td>NME</td><td>平均点误差 / 归一化距离</td><td>关键点精度</td></tr>
<tr><td>FID</td><td>‖μᵣ−μ_g‖² + tr(Σᵣ+Σ_g−2(ΣᵣΣ_g)^½)</td><td>生成图与真图的分布距离</td></tr>
<tr><td>3DMM</td><td>S = S̄ + A_id·α + A_exp·β</td><td>平均脸 + 形状 + 表情</td></tr>
<tr><td>弱透视投影</td><td>v = f·Pr·R·S + t₂d</td><td>3D 点到 2D 像素</td></tr>
<tr><td>WGAN-GP</td><td>D(x̃) − D(x) + λ(‖∇D(x̂)‖ − 1)²</td><td>稳定的 GAN 训练</td></tr>
</tbody></table></div>

### 答辩 / 面试标准答法 {#r-s2}

<div class="scroll"><table class="wide"><thead><tr><th>问题</th><th>标准答法（要点）</th></tr></thead><tbody>
<tr><td>人脸识别系统由哪几步组成？</td><td>检测（SCRFD）→ 关键点 → 相似变换对齐到 112×112 → 特征提取（ArcFace-ResNet）→ 特征归一化 → 余弦相似度比对（1:1 阈值 / 1:N 检索）。</td></tr>
<tr><td>为什么要对齐？</td><td>消除位置、尺度、旋转差异，让网络只关注身份；训练和测试必须用同一套对齐。</td></tr>
<tr><td>ArcFace 比 Softmax 好在哪？</td><td>特征和权重归一化后在超球面上比较角度；加性角度间隔强制类内紧凑、类间分离，测试时有安全余量；对困难样本给出更大梯度。</td></tr>
<tr><td>为什么用余弦而不是欧氏距离？</td><td>特征长度混有图像质量等无关信息；归一化后只比较方向，和训练目标一致（两者在单位球面上单调等价）。</td></tr>
<tr><td>开放集是什么意思？</td><td>测试的人不在训练集里，所以不能直接分类，必须学通用的特征度量。</td></tr>
<tr><td>LFW 为什么到不了 100%？</td><td>数据集本身有少量标注错误（同名不同人）；本项目剩余错误经人工检查绝大多数属于这类。</td></tr>
<tr><td>知识蒸馏为什么有效？</td><td>老师特征包含"谁和谁有多像"的暗知识，比 one-hot 标签信息量大；小数据上能显著提升学生并加快收敛。</td></tr>
<tr><td>量化为什么几乎不掉精度？</td><td>网络对小扰动鲁棒；逐通道权重量化 + 校准集确定激活范围，量化误差远小于特征的类间差距。</td></tr>
<tr><td>动态量化和静态量化区别？</td><td>动态：权重离线量化，激活运行时现算范围，只适合全连接/RNN；静态：用校准数据离线确定激活范围，卷积也能量化，速度提升明显。</td></tr>
<tr><td>如何保证实时性？</td><td>轻量模型（MobileNet 类）、降低检测分辨率或隔帧检测 + 关键点跟踪、量化、ONNX/推理引擎层融合、多线程流水线。</td></tr>
<tr><td>视频里关键点抖动怎么办？</td><td>One Euro 滤波：静止时强平滑去抖，快速运动时弱平滑降延迟。</td></tr>
<tr><td>3DMM 是什么？</td><td>用 PCA 从大量 3D 扫描中得到平均脸和形状、表情基，任意人脸 = 平均脸 + 基的线性组合；网络只需回归几十个系数。</td></tr>
<tr><td>StarGAN 怎么用一个模型做多种编辑？</td><td>把目标属性标签拼进生成器输入；判别器同时判真假和分类属性；循环一致性损失保证身份不变。</td></tr>
<tr><td>检测 AP 怎么算？</td><td>IoU≥0.5 算命中，扫分数阈值得 PR 曲线，求曲线下面积；WIDER 分 Easy/Medium/Hard，不属于该档的脸忽略。</td></tr>
</tbody></table></div>

### 常见失分点 {#r-s3}

1. BGR / RGB 搞反、HWC / CHW 搞反、归一化参数用错：程序不报错，准确率悄悄掉。每接一个新模型都要和参考实现逐元素对拍。
2. 训练和测试用不同的对齐方式（或不同的模板），准确率掉几个点。
3. 在测试集上选阈值、选 epoch、调超参数——结果虚高。
4. Umeyama 用 det(Σ) 判断镜像，点数少时会漏判。
5. ArcFace 的 θ + m 超过 π 后 cos 不再单调，需要线性兜底，否则会"奖励"错得离谱的样本。
6. NMS 阈值过低会误删合影里挨着的两张脸；检测阈值过高会漏掉被画面边缘切掉的脸（补边可解）。
7. 动态量化对卷积网络几乎无加速，要用静态量化；量化后一定要在验证集上重新测精度。
8. 非结构化剪枝后模型大小和速度都不变（稀疏矩阵仍按稠密存储计算），要么用结构化剪枝，要么配合稀疏推理引擎。
9. Windows 路径过长、OpenCV 5 不带 Haar 文件、ONNX 模型 batch 固定为 1——环境问题报的错往往不是真正原因。
10. 数据集人物与测试集重叠（如 MS-Celeb-1M 和 LFW 有重叠人物）会让准确率虚高，要做去重或换不重叠的训练集。
