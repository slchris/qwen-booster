# qwen-booster

**Apple Silicon 上混合注意力 Qwen 模型的实测推理优化与可复用原语。**

已在 M3 Max（40 核 GPU、128GB）上用 Qwen3.8-27B 与 Qwen3.8-Flash-Next 验证。
仓库里的每个性能结论都有 [`benchmarks/`](benchmarks/) 下冻结、带哈希的归档，
失败实验刻意保留。

[English](README.md) · [完整实验日志（中文）](docs/EXPERIMENTS.zh-CN.md) ·
[内网部署手册](docs/SERVING.md) · [证据下载](docs/EVIDENCE.md)

---

## 这是什么

- **一套实测工具**：长上下文解码曲线、并发基准、缓存正确性验证、开源测试集严格评分、
  零依赖 HTML 报告器。
- **可复用的原语**：vLLM 风格的块对齐混合前缀缓存、APFS 检查点磁盘缓存、
  确定性评分器，可直接移植到其他 MLX 项目。
- **一份诚实的记录**：有效与无效都保留（见下面状态表）。

## 这不是什么

- **不是推理框架**：服务路径是建立在 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
  固定版本之上的实验层。
- **不是通用加速**：多项优化与负载相关、默认关闭，且多为单次有界观测。
- **不是硬件矩阵**：两个模型 profile，一台机器类别。

## 状态

| 领域 | 状态 | 证据 |
| --- | --- | --- |
| 块对齐混合缓存（CPU + 27B 真机） | ✅ 已验证 | [hybrid-cache-native-v1](benchmarks/hybrid-cache-native-v1/manifest.json) |
| APFS 检查点磁盘存储 | ✅ CPU 测试 | `tests/test_apfs_store.py` |
| 长上下文解码（Flash，热态） | ⚠️ 有界 | 32K/100K/200K 分别 24.0 / 19.2 / 12.5–13.4 tok/s |
| 并发（连续批处理） | ⚠️ 有界 | [flash-concurrency-v1](benchmarks/flash-concurrency-v1/manifest.json) |
| 27B + DFlash2 草稿 | ⚠️ 有界（32K 1.77×） | [long-context-decode-scaling-v1](benchmarks/long-context-decode-scaling-v1/manifest.json) |
| 200K 热复用 | ⚠️ 依赖特定配方 | [flash-200k-disk-spill-accept-v1](benchmarks/flash-200k-disk-spill-accept-v1/manifest.json) |
| 冷后首问竞态修复 | 🧪 候选，因果未证 | [flash-200k-restore-race-candidate-v1](benchmarks/flash-200k-restore-race-candidate-v1/manifest.json) |
| Flash-Next KV 量化 | ❌ 上游拒绝 | QSA + 连续批处理 |
| 100K/200K 单流 30 tok/s | ❌ 未达到 | 需内核级优化 |

热态解码（M3 Max 128GB，Flash-Next 4-bit，greedy）：

| 上下文 | 解码 | 首 token（热） |
| ---: | ---: | ---: |
| 32K | 24.0 tok/s | 0.3s |
| 100K | 19.2 tok/s | 0.6s |
| 200K | 12.5–13.4 tok/s | 2–5s（需 200K 配方） |

200K 文档首次处理是批任务（Flash 约 5–12.5 分钟），不是交互操作。
分档配置与注意事项见 [docs/SERVING.md](docs/SERVING.md)。

## 快速开始（核心功能，无需模型）

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e .

python -m pytest -q                      # 252 个 CPU 测试，几秒

python -c "from qwen_booster.hybrid_cache import HybridPrefixCache"
python -c "from qwen_booster.grading import grade_answer, extract_number"

# 把测量 JSONL 渲染成离线 HTML 报告
PYTHONPATH=src python scripts/render_report.py results/bench.jsonl --output report.html

# 转换本地开源测试集导出（JSONL）
PYTHONPATH=src python scripts/build_opensource_suite.py --dataset mmlu \
  --input exports/mmlu.jsonl --output results/suite-mmlu.jsonl
```

服务路径需要独立固定环境与本地权重，见 [docs/SERVING.md](docs/SERVING.md)。

## 可复用原语

| 模块 | 用途 |
| --- | --- |
| `qwen_booster.hybrid_cache` | 块哈希链、对齐检查点、有界重放、LRU——面向注意力 KV 与循环/GDN 状态混合的模型 |
| `qwen_booster.apfs` | 临时文件 + 原子改名 + APFS `clonefile` 快照的检查点持久化 |
| `qwen_booster.grading` | 确定性评分：exact text、JSON、choice、numeric、contains、regex |
| `qwen_booster.runtime` | MLX-LM 单请求执行引擎 `Engine` / `HybridEngine` |
| `scripts/render_report.py` | 测量 JSONL → 离线 HTML 报告 |

## 仓库结构

```
src/qwen_booster/   核心模块与原生（MLX-LM）服务
scripts/            对外工具（基准、评测、报告）
scripts/experiments 原始调查中的一次性运行器
tests/              252 个 CPU 测试，无需 GPU
benchmarks/         冻结的证据归档（manifest + 原始输出 + 日志）
docs/               部署手册、完整实验日志、发布检查单
```

## 证据

证据目录不随源码仓库发布（约 200MB 原始输出、日志与 manifest）。**下载 Release 附件
并在仓库根目录解压**即可恢复 `benchmarks/`，见 [docs/EVIDENCE.md](docs/EVIDENCE.md)；
解压后本文档中的 `benchmarks/...` 链接才有效。每份 `manifest.json` 记录状态、说明、
产物哈希与归档时源码哈希；引用数字前先读 manifest。

## 已知缺口

- 100K+ 上到 30 tok/s 需要内核/编译层工作，固定版本的栈里没有现成开关。
- tools/函数调用与会话端点上游已有，但本项目未做端到端验收。
- 长时间 soak 与超过短测的多用户并发未覆盖。
- 公开发布前的检查单（含证据拆分方案）：[docs/RELEASE-CHECKLIST.md](docs/RELEASE-CHECKLIST.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方软件与模型许可见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
