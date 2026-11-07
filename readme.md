# D²NN 衍射光学神经网络实验

这个项目用 PyTorch 设计相位板。给定一束光和期望的输出，程序模拟光穿过各层后的传播，比较屏幕光强与目标，再用反向传播调整每个格子的相位。训练完成后，得到的是可以导出的相位图。仓库里的结果来自数值仿真。

## 两个光场设计实例

**光束整形：高斯光束变圆环。** 左图是目标光强，右图是光经过三块相位板后在屏幕上形成的圆环。

![圆环整形仿真，左侧为目标圆环，右侧为输出光强](docs/images/ring.png)

**固定图案投影：Bad Apple 剪影。** 左图指定人物和苹果的位置，右图显示同一束高斯光经过为这幅图案训练的相位板后的光强分布。

![固定图案投影仿真，左侧为目标剪影，右侧为输出光强](docs/images/silhouette.png)

[圆环的相位板](docs/images/ring-phases.png) · [剪影的相位板](docs/images/silhouette-phases.png)

## D²NN 擅长什么？

对于这里的单色、相干、纯相位 D²NN，最直接的任务是**控制光在空间里如何分布**。例如，输入中心最亮的高斯光，要求它在 120 mm 外形成圆环，目标就可以直接写成屏幕上的光强图。每张相位板改变光波各部分的相位，经传播和干涉，能量在不同位置重新分布。

| 任务 | 适用程度与原因 |
|---|---|
| 光束整形、固定图案投影 | 与器件的工作方式直接匹配。圆环和剪影实验分别把一束固定的高斯光变成指定图案。每换一个目标，当前脚本就要重新设计相位板。 |
| 空间滤波、光学模式变换 | 输入、输出都是光场，适合设计固定的空间变换。例如，可以针对高通滤波或特定输入模式的能量路由设计衍射层。[衍射层实现线性变换的研究](https://www.nature.com/articles/s41377-021-00623-5) |
| 类别固定、输入条件受控的图像识别 | 可以把不同类别的光送往不同探测区，以能量最大的区域作为预测。原始 D²NN 工作演示过手写数字分类；相机拍摄的照片还涉及光照、裁切、笔画粗细等分布变化。[D²NN 原始论文](https://doi.org/10.1126/science.aat8084) |
| 精确算术、条件分支、需要记忆的序列任务 | 对本项目的架构，这些通常不是优先选项。输入数字需要先编码成光场；固定相位板一路向前传播，没有保存上一步结果的状态。对于模 97 加法，直接计算 `(a + b) % 97` 就能对全部输入给出精确答案，而训练光学分类器还要检验未见组合上的正确率。 |

这种差别来自网络结构。相位板固定后，每层的相位调制和自由传播对**复光场**都是线性的，叠加多层仍得到一个线性变换；屏幕测量光强时才取模平方。更多层可以增加可实现的光场变换，但不会自动提供普通深度网络的逐层非线性、条件控制或记忆。对传统机器学习任务，是否合适取决于输入怎样编码、输出怎样读取，以及对新样本的泛化要求。[衍射网络的光学信息处理能力](https://pmc.ncbi.nlm.nih.gov/articles/PMC7844294/)

## 实验入口

| 目录 | 任务 | 运行方式 |
|---|---|---|
| [beam_shaping](beam_shaping/train.py) | 高斯光束 → 圆环 | 运行 `train.py`。完成后可运行 [replay_optics.py](beam_shaping/replay_optics.py)，用 SciPy 独立复算光路。 |
| [silhouette_shaping](silhouette_shaping/train.py) | 高斯光束 → bad apple | 将 [target.jpg](silhouette_shaping/target.jpg) 放在脚本旁，运行 `train.py`。默认让原图的黑色剪影发亮；可用 `BRIGHT_SILHOUETTE` 切换。 |
| [0-9classifier](0-9classifier/train.py) | MNIST 手写数字分类 | 在该目录运行 `train.py`，首次运行会按需下载 MNIST。测试自己的照片时，把 JPG 放进 `test/`，再运行 [predict_photos.py](0-9classifier/predict_photos.py)。 |
| [mod97_grokking](mod97_grokking/train.py) | 模 97 加法与延迟泛化实验 | 直接运行 `train.py`。当前版本自行生成全部 9,409 个有序数对，按约 50% / 25% / 25% 划分训练、验证和测试集。D2NN在该任务中没有优势。

## 如何运行

安装 Python 和依赖，在 IDLE 打开相应训练脚本，按 **F5**。圆环和剪影的脚本各自定义网络与参数，可以分开放置；移动剪影脚本时一并带上目标图片。训练优先使用 CUDA，实际设备会显示在控制台。

```bash
python -m pip install numpy matplotlib torch torchvision pillow scipy
```

也可以在项目根目录通过终端运行圆环实验：

```bash
python beam_shaping/train.py
python beam_shaping/replay_optics.py
```

两个整形实验都在各自的 `output/` 下保存 `weights.npz`（相位、配置和训练记录）、`phase_plates.png`（相位板合图），以及 `ring.png` 或 `result.png`（目标与输出）。再次完成训练会覆盖同名结果；比较不同种子或采样设置时，先留存上一组权重。


