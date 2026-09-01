# Shorkie PyTorch 非官方复现

> **Shorkie_LM 的非官方社区 PyTorch 复现。本项目与 Calico 或 Shorkie
> 原作者不存在隶属关系，也未获得其背书。**

本项目实现 16,384 bp 的掩码 DNA 语言模型，不是论文中包含 5,215 个
表达与表观组学轨道的八折监督 Shorkie。使用本项目时，请优先引用
[Shorkie 原论文](https://doi.org/10.1101/2025.09.19.677475)、访问
[官方仓库](https://github.com/calico/shorkie-paper)和
[官方文档](https://khchao.com/shorkie/)。

## 核心复现指标

![冻结 R64 验证合同上的总体加权困惑度](benchmarks/v0.1.0-rc1/overall_ppl.png)

| 模型 | 总体加权 PPL |
|---|---:|
| 作者官方 Shorkie_LM | 3.604430 |
| v1.1 D-best method-rebuild | 3.621104 |

PPL 越低越好。两者使用完全相同的 536 个重建 R64 valid 窗口、seed 165、
每窗 7 次前向、每次遮盖 2,457 个位置，以及相同的样本、mask 和反向互补
计划。D-best 比官方模型高 0.4626%，即略差；64 kb 配对区块 bootstrap
PPL ratio 的 95% CI 为 1.004244-1.005013。该结果不是作者原始语料上的
论文数值。

## 必须理解的边界

- 输入为 4 个 DNA 通道、1 个 mask 通道和 165 个物种通道，共 170 通道；
  输出为每个位点 4 个 A/C/G/T logits。
- 公开权重使用 Ensembl Fungi release 59 独立重建的 method-rebuild 语料，
  不等于作者原始语料。
- 若具备 GCP requester-pays 条件，应优先使用作者公开的
  `165_Saccharomycetales` 语料；本项目重建合同只是独立审计与备用路径。
- Figures 3-7 的真实监督性能尚未由本权重复现，MLM surprisal 不能冒充
  监督 Shorkie 的 `logSED`。
- 公开加载仅接受 `safetensors + config.json`；不要加载来源不明的
  PyTorch pickle 权重。

完整安装、下载、训练和评估命令见[英文 README](README.md)。数学合同、
数据结构、训练策略、证据和局限位于 `docs/`。

## 引用与联系

原 Shorkie 论文是首要引用。如果本 PyTorch 实现或 method-rebuild 权重
确实帮助了你的工作，可以再使用 [CITATION.cff](CITATION.cff) 附加引用
本版本。

**Ziyan Zhuang**  
Tianjin University；Shenzhen Loop Area Institute  
[ziyan@tju.edu.cn](mailto:ziyan@tju.edu.cn)  
[GitHub](https://github.com/ZiyanZhuang) · [Hugging Face](https://huggingface.co/ZiyanZhuang)

本项目采用 Apache-2.0 许可证。
