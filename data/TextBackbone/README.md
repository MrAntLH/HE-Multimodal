# 说明

此处数据和权重来自真实模型推理过程采集的中间数据。

权重：
	grounding-dino-base 模型，textbackbone部分bert模型权重，来自12层中的第1层；

输入：

```
text = ("woman with a white hat."
        "a gray elephant near green metal railing. "
        "a small water pool beside stone rocks. "
        "cute little baby in bright orange shirt.")
```

上述文本编码后产生的，共计 32 个 Token