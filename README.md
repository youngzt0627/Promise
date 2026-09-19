# Promise

基于 PyTorch 的生成式推荐模型实现。模型根据用户及历史交互信息，自回归预测三层语义 ID，并支持贪心和束搜索生成。

## 主要内容

- `model.py`：生成式推荐模型
- `amazon_data.py`：Amazon 数据集读取与特征适配
- `train_amazon.py`：训练和评估入口
- `self_check.py`：模型结构与生成流程自检

## 运行

安装 `torch` 和 `pandas` 后，可先执行自检，再启动训练：

```bash
python -m generative_recommender_impl.self_check
python -m generative_recommender_impl.train_amazon
```

Amazon 训练和测试数据未包含在仓库中，可通过命令行参数指定数据路径。

