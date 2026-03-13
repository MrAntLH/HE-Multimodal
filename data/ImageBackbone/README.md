# 数据说明：

数据采集自真实模型推理过程；

模型权重：

​	grounding-dino-base模型，ImageBackbone 部分 Swin 模型权重，Stage=2，Layer=1，即第三轮采样中第二层结构。

输入：

真实数据集，800×800像素（模型最小输入尺寸），推理至本层后，输入尺寸为:[50, 50, 512] （填充至[60, 60, 512]），

下一层为[25,25,1024]（填充至[36, 36, 1024]）

![image-20260125192817530](input_image.png)

注：

不同于Bert，Swin模型在自注意力模块输出时没有 LayerNorm

每 Stage 中每 Layer 结构包含：

​	LayerNorm_Before => Pad => Cyclic Shift => Self-Atten => Reverse Cyclic Shift => 去Pad => LayerNorm_After => FFN