# Qwen Booster

面向 Apple Silicon Mac 的 Qwen3.8-27B / Qwen3.8-Flash-Next 文本推理实验服务，基于 MLX / MLX-VLM 验证 prefill、MTP decode 与完整混合状态缓存。当前设备为 M3 Max、40 核 GPU、128GB 统一内存。

**当前尚未通过 200K 日常使用验收。** 最低目标为 204800 个真实输入 tokens，同文档连续追问和不同 200K 文档切换均为必测项。服务保留 262144 个总 tokens 的原生上限，输入、历史和输出共用预算；不截断或摘要输入来冒充支持。

交付目标为 **2026-10-07 20:00（北京时间）**。可运行代码与实测报告按此目标收敛，未通过项会明确保留。当前入口：

- [逐项验收表](benchmarks/acceptance-200k-v18.json)：已测范围、失败与剩余工作；早期 `acceptance.json` 仅是较短上下文阶段的历史审计。
- [工作负载证据汇总](benchmarks/workload-summary-v52.json)：88 份复核来源，保留实际缓存条件及限制。
- [交付计划](benchmarks/delivery-deadline-2026-10-07.json)：交付内容和时间目标，不能替代性能、质量验收。

新增多轮推理集成发现历史传递缺陷：第二轮应为 6900 tokens，后端消息格式化丢失思考字段后仅剩 6288 tokens。两次答案虽正确，本轮仍判失败并停止，[148 文件原始归档](benchmarks/flash-multiturn-reasoning-short-v1/manifest.json)。独立实验入口 `python -m qwen_booster.full_history_server` 已加入思考字段保留与 `preserve_thinking` 传递；两个模型共十组模板回归通过。首次修复重跑的五次回答正确且完整历史计数一致，但新增入口过早导入后端，导致缓存关闭，已停止并[归档失败记录](benchmarks/flash-multiturn-reasoning-short-v2/manifest.json)。入口已修正初始化顺序，模板和导入时预算检查通过；第三轮短集成已独立复核：实际输入 6156–9809 tokens，18/18 内容及格式正确、12/12 最新前缀命中、所有冷对照均零缓存；三组冷热生成文本不完全一致，不能作为相同输出提速结论。[151 文件归档](benchmarks/flash-multiturn-reasoning-short-v3/manifest.json)。同样三类任务的完整 200K 多轮测试已完成：实际输入 204818–208401 tokens，18/18 内容正确、17/18 裸 JSON 格式通过、12/12 最新前缀命中，所有冷对照均零缓存；首题初始回答的代码围栏格式失败保留。[161 文件归档](benchmarks/flash-multiturn-reasoning-200k-v3/manifest.json)。最终同历史冷／热耗时分别为配置题 532.50／41.71 秒、重试题 538.66／61.18 秒、调度题 523.94／40.56 秒。只有配置题的完整正文、思考及输出长度一致，其余两组存在生成差异；不能将这些缓存收益归因于单项优化或外推为稳定性能。热 decode 观测 15.33–16.64 tokens/s，报告 MLX 累计峰值最高约 105.48GB。Q8-27B 同类短集成已独立复核：6156–9004 个输入 tokens，18/18 内容与格式正确、12/12 最新前缀命中、三组最终冷热正文与思考完整一致，冷对照均零缓存。[144 文件逐字节归档](benchmarks/q8_27b-multiturn-reasoning-short-v3/manifest.json)。对应完整 200K 多轮已独立复核：204818–207961 个真实输入 tokens，18/18 内容与格式正确、12/12 最新前缀命中、三组最终冷热正文、思考与输出长度完全一致，冷对照均零缓存。[166 文件逐字节归档](benchmarks/q8_27b-multiturn-reasoning-200k-v3/manifest.json)。这是三个合成任务族的单次有界验证，不代表广泛日常质量或稳定性能通过。

<details>
<summary>保留完整思考历史的当前启动方式</summary>

在项目根目录运行，使用下文锁定的 VLM 环境和本地模型目录。以下参数直接取自本轮测试配置；27B 的模型路径需替换成本机 Q8 目录。两款模型共用端口 18083，应分别启动；当前测试运行期间不要额外启动模型进程。

Flash-Next：200K 多轮已完成有限验证：

```bash
PYTHONPATH=src MLX_GDN_PACKED=1 APC_CHECKPOINT_ENTRIES=4 APC_MEMORY_RESERVE_GB=3.725290298461914 APC_DISK_ENABLED=0 APC_CHECKPOINT_INTERVAL_TOKENS=0 \
.cache/vlm-env/bin/python -m qwen_booster.full_history_server --model .cache/models/qwen3.8-flash-next-4bit-external-ple --context 262144 --port 18083 --cache-mb 24000 --memory-mb 110000 --allocator-cache-mb 2000 --token-timeout 7200 --draft .cache/qwen3.8-flash-next-mtp-4bit --release-finished-mtp --borrow-flash-mtp-checkpoint-row --credit-flash-mtp-live-warm-cache --recheck-flash-mtp-cache-admission --recheck-flash-mtp-cache-restore
```

27B Q8：短集成与 200K 多轮均已完成有限验证：

```bash
PYTHONPATH=src MLX_GDN_PACKED=1 APC_CHECKPOINT_ENTRIES=4 APC_MEMORY_RESERVE_GB=3.725290298461914 APC_DISK_ENABLED=0 APC_CHECKPOINT_INTERVAL_TOKENS=0 \
.cache/vlm-env/bin/python -m qwen_booster.full_history_server --model /path/to/Qwen3.8-27B-MLX-8bit --context 262144 --port 18083 --cache-mb 48000 --memory-mb 110000 --allocator-cache-mb 2000 --token-timeout 7200 --draft .cache/qwen3.8-mtp-4bit --release-finished-mtp --fused-prefill --borrow-27b-checkpoint-row --credit-27b-live-warm-cache --recheck-27b-cache-admission
```

客户端调用 `/v1/chat/completions` 时，在请求顶层设置 `"preserve_thinking": true`。后续轮次发送完整 `messages`：保留上一轮 assistant 的 `content` 和实际返回的 `reasoning_content`，再追加用户消息；流式客户端需要分别累积两类字段。不要只回传正文，否则服务无法恢复客户端丢弃的思考历史。输入、历史与预留输出共同受 262144-token 上限约束。

启动后查看 `/v1/cache/stats`：`enabled` 应为 `true`，27B 的 `memory_max_bytes` 应为 48000000000，Flash 应为 24000000000；两组配置均关闭磁盘缓存。这些是同文档完整历史配置，双文档磁盘切换的已有证据与参数见下文；新增历史入口与磁盘切换的组合尚未完成独立验证。

</details>

两模型完整历史短测的[逐请求对比](benchmarks/multiturn-model-comparison-short-v1.json)已汇总 18 组对应观测：仅三道初始问题的完整输入相同，后续历史包含各模型实际生成的正文和思考，输入会分叉。表内分别记录输入是否一致、质量、首个思考／正文时间、总耗时、decode 和缓存状态；不将这些差异归为单项优化收益，也不外推到 200K。

两模型完整历史 **200K** 的[逐请求对比](benchmarks/multiturn-model-comparison-200k-v1.json)已完成，包含 18 组对应观测。以下为当前 M3 Max 128GB 上的观测区间，decode 包含思考 tokens，耗时不含模型加载：

| 200K 多轮指标 | 27B Q8 组合 | Flash-Next 组合 |
| --- | --- | --- |
| 首轮冷 prefill，tokens/s | 98.12–101.09 | 415.52–425.21 |
| 首轮冷 decode，tokens/s | 8.94–9.60 | 15.08–15.38 |
| 命中缓存的追加追问 decode，tokens/s | 9.49–10.38 | 15.33–16.64 |
| 首轮冷请求总耗时 | 2119.83–2187.83 秒 | 541.27–569.74 秒 |
| 追加追问总耗时 | 77.94–116.62 秒 | 43.81–77.88 秒 |
| 最终相同历史缓存重放总耗时 | 60.87–86.64 秒 | 40.56–61.18 秒 |
| 内容／裸 JSON 正确 | 18/18 ／ 18/18 | 18/18 ／ 17/18 |
| 最新前缀命中 | 12/12 | 12/12 |
| 最终冷热完整输出一致 | 3/3 | 1/3 |
| 报告的 MLX 累计峰值 | 98.31GB | 105.48GB |

只有三道首轮问题的完整输入完全相同；追加历史包含各自实际回答和思考，因而分叉。模型、量化、内存预算和运行时段不同，不能把表中差距写成单项加速百分比。Flash 在这组任务中更快，可优先试用于延迟敏感的长文档推理；27B Q8 在这组任务中格式及最终冷热输出一致性更好，不能据此认定整体更聪明。两个模型首次处理 200K 仍以分钟计，均未达到所有日常场景流畅的目标。

当前最重要的结果：

27B Q8 首个完整历史 200K 冷请求的瓶颈已单独归档：204842 个输入 tokens，总耗时 2119.83 秒，其中 prefill 2026.31 秒，占 95.6%。日志内两个相同长度的 38912-token 区间，前段处理速度约 160.12 tokens/s，后段约 71.55 tokens/s。它说明长输入后段处理明显变慢，但单次日志不能确定是注意力、GDN、内存访问还是温度因素，也不能据此宣称某项优化有效。[分析、不可变日志快照与来源哈希](benchmarks/q8-200k-first-cold-prefill-analysis-v1/manifest.json)。

| 配置 / 场景 | 已有证据 | 尚未通过的部分 |
| --- | --- | --- |
| 27B Q4 组合优化，200K 双文档十轮交替 | 28/28 回答正确、24/24 最新前缀命中；三次按原内存门槛复查恢复保存 | 重复检索诊断，不能代表广泛代码质量或稳定性能 |
| 27B Q4，261265-token 原生边界样例 | 冷请求及三次热请求均完成，三次完整前缀命中 | 四次答案均错误；Q8 同题两配置共四次答对，组合重复请求命中 261264 tokens；仅单题证据 |
| Flash MTP，保存准入复查，200K 十轮会话 | 10/10 正确、9/9 最新前缀命中；7 次按原门槛恢复保存，末轮冷热答案一致 | 四道轮换检索题，广泛会话推理仍未验收 |
| Flash MTP，恢复复查候选，200K 双文档十轮交替 | 28/28 内容及格式正确、24/24 最新前缀命中；四组冷热一致，21 次 SSD 恢复 | 本轮未触发恢复重试，不能证明该修复有效；广泛推理、最大上下文与稳定性能仍待验收 |

Q8-27B 的未加速／组合优化短对照已独立复核：12/12 内容及格式正确，六组冷热答案、思考及输出长度一致，MTP 运行计数已确认。三题跨配置答案相同，但两题思考文本与长度不同，因此不以总耗时差直接宣称同输出提速。[136 文件归档](benchmarks/q8-27b-combined-short-v1/manifest.json)。完整 261265-token 历史错误样例的两配置冷热对照已独立复核：四次均答对，正文、思考及输出长度一致。未加速首次／重复请求为 3918.12／3905.45 秒，均未命中缓存；组合首次为 3341.12 秒，重复请求命中 261264 tokens，仅需 4.93 秒，首个可见回答 2.50 秒。冷 prefill 为 66.72 → 78.28 tokens/s；这是一次有序对照，组合冷请求耗时减少约 14.7%，不能归因于某一个开关或代表稳定收益。仅输出 9 tokens，decode 观测反而由 6.63 降至 3.15 tokens/s，不作为持续生成基准。组合冷／热报告 MLX 峰值 69.15／81.43GB，后者可能含进程累计峰值。[135 文件逐字节归档](benchmarks/q8-27b-combined-native-v1/manifest.json)。单题通过不代表最大上下文广泛质量通过。


当前 Flash 组合配置的三类 200K 跨文件推理对照已完成：输入 204818–204842 tokens，每题生成 712／1248／1211 tokens，两配置共 12/12 内容正确、8/12 裸 JSON 格式通过。首题两配置的冷热回答均带代码围栏，格式失败保留。六次重复请求均命中完整前缀；六组冷热及三组跨配置的正文、思考和输出长度一致。[152 文件逐字节归档](benchmarks/flash-current-reasoning-200k-v1/manifest.json)。

| Flash 推理任务 | 冷请求秒：未加速 → 组合 | 缓存请求秒：未加速 → 组合 | 冷 decode tokens/s：未加速 → 组合 |
| --- | ---: | ---: | ---: |
| 配置优先级 | 573.49 → 545.78 | 52.59 → 50.33 | 13.46 → 14.31 |
| 重试计数 | 603.23 → 602.94 | 91.36 → 88.93 | 13.50 → 14.29 |
| 依赖调度 | 589.91 → 614.35 | 89.17 → 79.86 | 13.42 → 14.44 |

这是同权重、同输入、同内存预算的单次有序对照；冷请求表现有改善、持平和回退，不构成稳定普遍加速。报告 MLX 峰值最高为未加速 97.44GB、组合 99.16GB，含同进程累计峰值的可能；首个思考和首个可见答案延迟在原始复核中分开记录。有限合成推理题不代表所有日常任务；Flash 原生边界前／中／后位置检索对照已完成：实际输入 262015–262016 tokens，12/12 回答正确，六组冷热和三组跨配置正文、思考及输出长度一致；基线完整前缀命中 2/3，组合配置 0/3，组合原生上限缓存保留未通过。[157 文件逐字节归档](benchmarks/flash-current-native-boundary-v1/manifest.json)。

原生边界三位置的冷请求总耗时为基线 717.61／719.15／714.44 秒，组合 717.66／708.30／707.63 秒。基线前／后重复请求为 4.02／3.63 秒，中部未命中为 714.36 秒；组合三次重复请求均未命中，需 704.27／706.87／705.33 秒。仅输出 14–15 tokens，不作为持续 decode 基准；一次有序对照不能宣称稳定冷启动提速。组合三处现场均显示零次成功保存、零次驱逐和两次内存准入跳过；现有保存复查要求实际驱逐，未覆盖这一首次保存路径。同预算诊断已完成：两次答案正确且与原始对照一致，重复请求仍零命中、耗时 693.80 秒。四次拒绝均发生在完整快照保存的内存准入检查；缓存容量足够、没有驱逐，检查后的内存余量采样比要求少 1.71–2.87GB。日志采样与实际保护条件的读取并非同一时刻，不据此宣称精确瞬时差值或计时收益。[138 文件逐字节归档](benchmarks/flash-native-first-store-diagnostic-v1/manifest.json)。保持原内存保护、24GB RAM 缓存和新建 48GB SSD 缓存的前部样例回退测试已独立复核：RAM 保存被拒后同步写入约 7.587GB，重复请求实际从 SSD 恢复 262015 tokens；两次回答正确且一致，首次／重复请求 711.85／6.28 秒，磁盘写入失败为零。[138 文件逐字节归档](benchmarks/flash-native-ssd-spill-v1/manifest.json)。这只证明一个样例的回退与恢复可行，中／后位置、广泛推理和持续可靠性仍待验证；带诊断的单次结果不作为稳定性能承诺。 中部位置补测已独立复核并[归档 140 个文件](benchmarks/flash-native-ssd-spill-middle-v2/manifest.json)：262016 个真实输入 tokens，两次答案正确且一致，冷／热请求 751.28／3.73 秒，热命中 262015 tokens。本次 RAM 保存成功，另有一次 SSD 写入、零写入失败，但实际 disk_hits 为 0；不能据此认定中部 SSD 恢复已验证，或将未触发回退写成回退失败。末段补测也已[独立复核并归档 140 个文件](benchmarks/flash-native-ssd-spill-end-v2/manifest.json)：262015 个真实输入 tokens，两次答案正确且一致，冷／热请求 707.00／4.01 秒，热命中 262014 tokens；同样为 RAM 命中，SSD 写入一次、实际恢复零次。[三位置汇总](benchmarks/native-ssd-positions-v1.json)保留全部六次正确回答，只有前部位置实际触发同步回退和磁盘恢复。中／末位置的新进程恢复现已独立复核：启动时 RAM 缓存为空，两处分别实际从 SSD 恢复 262015／262014 tokens，请求耗时 12.71／11.72 秒，随后 RAM 重复请求 3.63／3.37 秒；四次回答均正确且与原冷请求正文、思考、长度一致，磁盘写入失败为零。[中段 148 文件归档](benchmarks/flash-native-restart-restore-middle-v1/manifest.json)、[末段 148 文件归档](benchmarks/flash-native-restart-restore-end-v1/manifest.json)。三个位置均有实际 SSD 恢复证据，但 RAM 被拒后同步回退仍只在前部触发；上述单次诊断不证明广泛推理质量或持续可靠性。

Flash MTP 未启用准入复查的追加历史测试**未通过**：实际输入 204801–205133 tokens，前 8 次回答内容正确、7 次最新前缀命中；第 8 次回答后，旧快照已淘汰、新快照保存被跳过，缓存驻留归零。测试按规则停止，后两轮及最终冷回放未执行。[失败复核与原始归档](benchmarks/flash-mtp-credit-conversation-200k-v1/manifest.json)。相同提示连续复用通过，不能替代追加历史保留验收。

针对上述失败的 `--recheck-flash-mtp-cache-admission` 候选默认关闭，只在实际淘汰后保存被拒绝时按原容量和内存余量门槛复查，最多 250 毫秒。短测 12/12 内容与格式通过，冷热及跨配置输出一致。[短测归档](benchmarks/flash-mtp-recheck-short-v1/manifest.json)。随后完整 204801–205227-token 十轮重放已独立复核：10/10 正确、9/9 最新前缀复用，每轮最新快照均保存；7 次实际淘汰后拒绝通过原门槛复查恢复，零次复查失败。热首字中位数 1.98 秒；末轮无缓存对照 579.02 秒、答案与热请求一致。前八轮完整输入及输出与失败对照一致。[125 文件逐字节归档](benchmarks/flash-mtp-recheck-conversation-200k-v1/manifest.json)。这是四个轮换检索问题，不能替代广泛推理质量；双文档 RAM＋SSD 十轮交替随后未通过：已完成 17/17 内容与格式正确，15 次最新前缀检查中 14 次通过；B 第五轮恢复准入被拒后全量重算 598.43 秒，第七轮同类失败重现，已主动停止并保留未完成轮次。[127 文件失败归档](benchmarks/flash-mtp-recheck-disk-switching-200k-v1/manifest.json)。恢复阶段的独立候选 `--recheck-flash-mtp-cache-restore` 默认关闭，要求先启用保存准入复查；原内存门槛不变，仅在实际淘汰后拒绝时复查，最多 250 毫秒。CPU 保护检查与真实模型短对照均已通过：12/12 内容及格式正确，6 组冷热和 3 组跨配置输出一致，但零次实际恢复重试。[140 文件短测归档](benchmarks/flash-mtp-restore-short-v1/manifest.json)。完整双文档十轮交替现已复核：28/28 内容及格式正确、24/24 最新前缀命中，四组完整历史及输出一致；21 次 SSD 恢复，零次跳过保存。末轮 SSD／RAM 请求分别 34.00／7.39 秒，对应独立进程、关闭磁盘缓存的冷对照为 640.27／613.96 秒。20 次保存准入复查实际恢复，但恢复阶段复查零次触发，因此新恢复修复的因果有效性仍未验证。该重复问题诊断也不代表广泛推理质量或稳定提速。[132 文件逐字节归档](benchmarks/flash-mtp-restore-disk-switching-200k-v1/manifest.json)。

最新双文档任务的同输入速度观测如下（204807–205849 输入 tokens，每次输出 75 tokens）。已逐条核对两模型 28 份完整输入的消息与 token 哈希一致；缓存预算、模型和运行时间不同，不是单项优化的因果对照。[可复现对比数据](benchmarks/current-switching-comparison-v1.json)。

| 配置 / 缓存条件 | 观测次数 | 冷 prefill tokens/s | decode tokens/s | 整次请求耗时 |
| --- | ---: | ---: | ---: | ---: |
| 27B Q4 组合，冷请求 | 4 | 94.00–101.98 | 6.40–10.39 | 2015.99–2200.88 秒 |
| 27B Q4 组合，RAM 命中 | 24 | — | 7.99–10.52 | 8.77–13.85 秒 |
| Flash MTP 候选，冷请求 | 4 | 325.90–363.08 | 10.76–13.22 | 571.78–640.27 秒 |
| Flash MTP 候选，RAM 命中 | 3 | — | 12.54–13.75 | 6.84–9.55 秒 |
| Flash MTP 候选，SSD 命中 | 21 | — | 10.67–14.73 | 31.25–43.71 秒 |

各行是本轮观测范围，不是置信区间或稳定性能承诺；75-token 短答案不足以确定持续 decode 吞吐。27B／Flash 的最高报告 MLX 峰值为 84.37／105.23GB，可能包含同进程先前请求的累计峰值。

以上单次/有界观测不构成稳定提速或普遍“不降智”结论。约 200K 输入的 Flash MTP 原配置在两项仓库任务中，decode 仅有约 1%–10% 的单次增长，冷请求总时间反而增加，重复请求还未命中缓存；短输入的约 55%–60% 热 decode 增长不能外推到长上下文。首个思考 token、首个正文 token、完整回答时间分开记录。

[27B 交替会话归档](benchmarks/recheck-27b-switching-conversation-200k-v1/manifest.json) · [27B 原生边界失败](benchmarks/credit-27b-261k-admission-v1/manifest.json) · [Flash 200K 原始对照](benchmarks/flash-next-mtp-repository-200k-v1/manifest.json) · [借用快照重放](benchmarks/flash-mtp-borrow-200k-v1/manifest.json) · [恢复计量短测](benchmarks/flash-mtp-credit-short-v1/manifest.json) · [200K 连续复用归档](benchmarks/flash-mtp-credit-200k-v1/manifest.json)

下面两组是**当前实验配置的启动方式**，优化开关仍默认关闭。固定模型及依赖准备、原生 API 与测量说明见后文；同机 GPU 实验需串行运行。

```bash
# 27B 组合候选：完整上下文、48GB RAM 缓存、末尾检查点。
PYTHONPATH=src MLX_GDN_PACKED=1 APC_CHECKPOINT_ENTRIES=4 \
APC_CHECKPOINT_INTERVAL_TOKENS=0 APC_MEMORY_RESERVE_GB=3.725290298461914 APC_DISK_ENABLED=0 \
.cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model /path/to/Qwen3.8-27B-4bit --draft .cache/qwen3.8-mtp-4bit \
  --context 262144 --port 18083 --cache-mb 48000 --memory-mb 110000 \
  --allocator-cache-mb 2000 --token-timeout 7200 --release-finished-mtp \
  --fused-prefill --borrow-27b-checkpoint-row --credit-27b-live-warm-cache \
  --recheck-27b-cache-admission

# Flash MTP 单文档候选：24GB RAM 缓存，十轮有限检索已通过；非双文档 SSD 实验配置。
PYTHONPATH=src MLX_GDN_PACKED=1 APC_CHECKPOINT_ENTRIES=4 \
APC_CHECKPOINT_INTERVAL_TOKENS=0 APC_MEMORY_RESERVE_GB=3.725290298461914 APC_DISK_ENABLED=0 \
.cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model .cache/models/qwen3.8-flash-next-4bit-external-ple \
  --draft .cache/qwen3.8-flash-next-mtp-4bit --context 262144 --port 18083 \
  --cache-mb 24000 --memory-mb 110000 --allocator-cache-mb 2000 \
  --token-timeout 7200 --release-finished-mtp --borrow-flash-mtp-checkpoint-row \
  --credit-flash-mtp-live-warm-cache \
  --recheck-flash-mtp-cache-admission
```


频繁切换两份 200K 文档时，上述 Flash 单文档命令需将 `--cache-mb 24000` 改为 `--cache-mb 32000`，并追加：

```bash
--disk-cache-dir .cache/apc-flash-daily --disk-cache-mb 48000 \
--recheck-flash-mtp-cache-restore
```

这是已归档双文档实验的 32GB RAM＋48GB SSD 缓存预算；启动器根据磁盘参数设置 `APC_DISK_ENABLED`。首次使用新目录仍需完整 prefill，后续命中分别按 RAM 或 SSD 的实际延迟计；两份文档总计约 400K tokens 不表示单次请求支持 400K。归档实验另启用了准入诊断日志，以上常规启动方式的稳定性能尚未单独验收。具体环境和命令见[双文档实验配置](benchmarks/flash-mtp-restore-disk-switching-200k-v1/results/flash-mtp-restore-disk-switching-200k-v1-config.json)。

<details>
<summary>按实验顺序保留的历史记录与限制</summary>

当前目标以至少 204800 个真实输入 tokens、两款模型、同文档多轮与双文档切换为准。[200K 验收表](benchmarks/acceptance-200k-v18.json)逐项列出证据与缺口；早期 `benchmarks/acceptance.json` 是较短上下文阶段的历史审计，不表示当前日常使用目标已完成。交付目标为北京时间 2026-10-07 20:00，未通过项将明确保留。

Flash-Next MTP 最新实测（2026-10-06）：完整 206783 / 206632-token 两项仓库任务、两种配置、冷热共 8 条响应已独立复核，全部内容正确、裸 JSON 0/8，跨配置答案、推理和输出长度一致。冷请求 decode 分别由 12.92 → 14.26、12.71 → 12.87 tokens/s；但 MTP 冷请求总耗时分别增加 1.3% / 11.7%，且两项重复请求均缓存命中 0，耗时 1164.53 / 987.88 秒，对应无 MTP 热缓存为 348.50 / 137.36 秒。**这组原始 Flash MTP 配置未通过 200K 缓存复用验收；后续候选的有限通过范围见上方，日常长上下文整体验收仍未完成**。每条件仅一次，不能作稳定提速结论。[完整复核](benchmarks/flash-next-mtp-repository-200k-v1/results/flash-next-mtp-repository-200k-v1-review.json)与[141 文件逐字节归档](benchmarks/flash-next-mtp-repository-200k-v1/manifest.json)。此前短输入 12/12 内容与格式通过、热 decode 单次提升约 55%–60%，不外推到 200K。[短输入归档](benchmarks/flash-next-mtp-http-short-v1/manifest.json)。机器可读[汇总 v50](benchmarks/workload-summary-v52.json)已纳入短测、200K 对照和准入诊断，共 88 份复核来源。

Flash MTP 缓存失败的诊断已完成：206632-token 原题冷请求及重复请求的四次快照保存均因运行时内存余量不足被拒绝；24GB 缓存容量足够，但所需余量约 16.03GB，实际仅 9.83–11.38GB。[127 文件诊断归档](benchmarks/flash-next-mtp-admission-diagnostic-v1/manifest.json)。减少快照提取复制的 `--borrow-flash-mtp-checkpoint-row` 候选默认关闭，GPU 状态与快照隔离探针已通过，真实 MTP 短测已复核：12/12 内容与裸 JSON 通过，六组冷热及三组跨配置答案、推理与输出长度一致，候选日志确认实际使用新路径。[短测归档](benchmarks/flash-mtp-borrow-short-v1/manifest.json)。随后完整 206632-token 冷/热重放已完成：两次内容正确，正文、推理、输出长度与原对照一致；热请求命中 206631 tokens，总耗时 130.96 秒。但恢复阶段发生一次内存淘汰，缓存驻留归零，持续多轮保留仍未通过。[132 文件重放归档](benchmarks/flash-mtp-borrow-200k-v1/manifest.json)。新的恢复预留计量候选 `--credit-flash-mtp-live-warm-cache` 默认关闭；短测 12/12 内容与格式通过、六组冷热及三组跨配置输出一致，六次恢复抵扣均通过上限复核。[短测归档](benchmarks/flash-mtp-credit-short-v1/manifest.json)。完整 206632-token 冷请求加三次重复请求已复核：4/4 内容正确、0/4 裸 JSON，三次命中 206631 tokens，约 6.01GB 快照始终保留，零淘汰、零跳过；热总耗时 128.71–129.72 秒，decode 14.24–14.34 tokens/s。正文、推理及输出长度与原对照一致。[136 文件归档](benchmarks/flash-mtp-credit-200k-v1/manifest.json)。这是相同提示重复，追加历史和双文档切换仍需单独验证。

当前验收状态（2026-10-06）：**尚未完成日常使用验收**。最新组合栈的真实仓库任务计划 8 次响应，实际完成 6 次：内容 6/6 正确、裸 JSON 2/6；206632-token 任务重复请求缓存命中为 0，触发停止，该轮未执行最后一组 MTP 基线，后来已在下述独立诊断中补测。失败记录已[独立复核并完整归档](benchmarks/combined-27b-repository-v1/manifest.json)，此前检索任务的成功不能替代这项失败。

针对空缓存首次保存拒绝的新候选 `--recheck-27b-first-store` 默认关闭，要求同时启用 `--recheck-27b-cache-admission`，沿用原内存门槛和最多 250 毫秒的有界复查。三道短任务的两种配置共 12 次响应已复核：内容及裸 JSON 12/12 正确，六组冷热一致、三组跨配置输出一致，但没有触发拒绝重试，**尚不能证明修复有效**。[短测试归档](benchmarks/first-store-27b-http-short-v1/manifest.json)。相同 48GB 缓存预算的完整 206632-token 诊断重放及缺失基线已完成：4/4 内容正确、2/4 裸 JSON 合格，两组冷/热缓存复用通过，但原始准入均直接成功，零次首次保存重试，修复有效性仍未证明。[诊断归档](benchmarks/first-store-27b-repository-replay-v1/manifest.json)。Flash-Next MTP 已完成上方短测和 200K 对照，但长上下文缓存复用仍未通过；原生上限 Q4 质量问题仍未解决。

最新准入复查候选已完成 28 请求独立复核：全部输入 204807–205849 tokens，完整历史重新分词通过，28/28 答案及裸 JSON 正确、24/24 最新前缀命中、四组冷热输入与输出一致。三次淘汰后的保存拒绝在 59–66 毫秒内按原内存门槛复查成功，零次跳过保存；此前对照为 21/24 命中、三次跳过。最终 A/B 热访问首字 1.78/1.73 秒、总耗时 11.04/8.77 秒；同历史冷对照 2091.30/2200.88 秒，prefill 98.99/94.00 tokens/s，进程累计 MLX 峰值 84.37 GB。这是两份资料各十轮重复检索的单次带日志诊断，不代表广泛质量、稳定提速或原生上限质量通过，候选仍默认关闭。[独立复核与完整归档](benchmarks/recheck-27b-switching-conversation-200k-v1/manifest.json)。

此前未启用准入复查的双文档十轮交替诊断：28/28 回答及裸 JSON 正确，所有完整历史重新分词通过，实际输入 204807–205849 tokens，四组冷热完整提示与输出一致。但最新前缀复用仅 **21/24**：B 第 2、3 轮与 A 第 5 轮回退到旧前缀，对应三次跳过快照保存，**持续缓存保留未通过**。最终 A/B 热请求耗时 10.48/8.61 秒、首字均约 1.72 秒；同历史清缓存对照耗时 2008.58/2035.63 秒、prefill 103.02/101.57 tokens/s。进程累计 MLX 峰值约 84.37 GB。本轮带准入日志、每条件一次，且反复回答同一检索题，不能作为受控提速、稳定 decode 或广泛多轮质量结论。[独立复核与完整归档](benchmarks/credit-27b-switching-conversation-200k-v1/manifest.json)。

双文档缓存诊断补充：相同组合配置的带内存日志复测已结束，8/8 回答与格式正确，两个追加历史重访分别精确命中 204907 / 204939 tokens。四个快照保存成功，发生两次旧快照淘汰、零次跳过保存，因此没有复现上轮 B 追问快照保存失败。两次预填充准备阶段的准入返回 false，但随后采样的余量已恢复；这些非同时采样不能直接说明上一轮失败原因。持续切换稳定性仍未验收，本轮仅用于诊断，不作为受控性能对照。[独立复核与完整归档](benchmarks/credit-27b-switching-admission-200k-v1/manifest.json)。

最新组合候选（融合 Attention＋MTP＋完整前缀缓存＋借用单行视图＋已恢复缓存预留计量修正）的 200K 十轮会话已完成复核：实际输入 204801–205227 tokens，10/10 回答正确，九次追加均复用上一轮输入长度减一的前缀，热追问首个可见回答中位 3.57 秒；最终完整历史清空缓存重放 2050.01 秒，prefill 100.24 tokens/s，答案及 reasoning 一致。进程累计 MLX 峰值约 56.96 GB。这仍是一份合成资料中的四道循环检索题，输出仅 9–12 tokens，不能作为稳定 decode 或广泛质量结论；与历史运行的时间差不是受控提速比例，候选默认关闭。[复核与完整归档](benchmarks/credit-27b-conversation-200k-v1/manifest.json)。

同一组合候选的双 200K 文档纯 RAM 切换检查已完成：8/8 回答和裸 JSON 格式正确，四组完整提示及输出一致，四次热请求复用原文档完整前缀。两份新文档首次请求分别约 33.75 / 34.10 分钟，prefill 101.52 / 100.51 tokens/s；切回原文档耗时 9.25 / 9.57 秒，追加追问耗时 12.80 / 13.55 秒。同历史清空缓存追问分别需 33.91 / 33.86 分钟。75-token 输出的 decode 观测范围 8.09–10.24 tokens/s，进程累计 MLX 峰值约 84.06 GB。但 B 追问结束时出现一次淘汰和一次跳过快照保存，持续交替追问的缓存保留尚未通过；正在另行诊断，不能以本轮四次命中替代持续保留验收。每种条件仅一次，同进程清缓存对照。[复核与完整归档](benchmarks/credit-27b-switching-200k-v1/manifest.json)。

原生上限附近的历史错误样例也已复测：261265-token 完整输入的冷请求及三次重复热请求均正常完成，三次热请求均命中 261264 tokens，约 17.28 GB 快照持续保留且无淘汰或跳过保存。然而四次答案仍为错误的 `30158402`，正确答案应为 `97804441`。因此这项结果只证明该样例的缓存保留修正有效，**最大上下文质量仍未通过**。[诊断与完整归档](benchmarks/credit-27b-261k-admission-v1/manifest.json)。下方历史实验保留其各自配置与限制；机器可读汇总 v49 保留上述失败及 88 份复核来源。

融合 Attention＋MTP＋完整前缀缓存的 200K 十轮会话已完成复核：实际输入从 204801 增至 205227 tokens，10/10 回答正确，九轮追加均复用上轮输入长度减一的前缀，首个可见回答中位 4.04 秒。最终完整历史清空缓存重放耗时 2106.17 秒，答案及 reasoning 与热请求一致。四轮短会话冷/热一致性检查也通过。这是一份资料中的四道循环检索题，不代表十道独立任务、稳定 decode、P95 或最大上下文质量通过；优化仍默认关闭。[完整归档](benchmarks/fused-mtp-conversation-200k-v1/manifest.json)。

HTTP＋MTP＋完整前缀缓存的融合注意力集成对照已完成：三道题、短/200K 两种长度、基线/优化两模式及冷/热，共 24 次响应；内容和裸 JSON 均 24/24 正确，12 组冷/热正文与 reasoning 一致，热请求全部命中输入长度减一的完整前缀。三道 200K 题 prefill 从 72.49/71.96/80.17 提升到 89.98/92.37/92.62 tokens/s，单次观测提升 24.12%/28.36%/15.53%；整题耗时下降 19.02%/20.83%/12.76%，优化版仍需 38.33–39.07 分钟。Decode 无稳定改善，进程累计 MLX 峰值约 59GB、基本不变；第二题优化前后思考长度从 1076 变为 1153 tokens，答案相同。每种条件仅一次、基线先行，不能作稳定提速或广泛质量结论，默认关闭；组合配置的多轮、多文档和原生边界仍待验证。[原始响应、冻结源码与复核归档](benchmarks/fused-http-mtp-v1/manifest.json)。

融合注意力 200K 重复对照已完成：同一份 204842-token 原题，按 AB/BA/AB/BA 顺序执行四组，共八次回答与裸 JSON 全部正确，每组输出 token 完全一致。配对 prefill 吞吐提升 4.32%–10.37%，中位数 9.06%；整题耗时下降中位数 8.06%，峰值 MLX 内存下降 38.65%。decode 变化中位数为 -4.75%，没有一致改善。单进程、无预热、仅一道独立题，绝对速度存在波动；该独立模型重复实验不覆盖 HTTP/MTP/完整前缀缓存组合；最大上下文质量尚未验收，默认不开启。[重复对照归档](benchmarks/fused-tasks-200k-repeated-v1/manifest.json)。


融合注意力独立生成的短任务筛查已完成：三道原题共六次响应，答案与裸 JSON 格式全部正确；两题生成 token 不同，一题完全一致。首 token 提速比分别为 0.977 / 1.020 / 0.979，未显示明显收益。每题只有一组基线先行的冷对照，没有丢弃预热或平衡顺序，不作为稳定性能结论。[短任务归档](benchmarks/fused-tasks-short-v1/manifest.json)。三道完整 204818–204842-token 原题的独立生成对照已完成：六次答案和裸 JSON 均正确，三组输出 token 完全一致。Prefill 为基线 86.88–88.37、优化版 94.48–96.51 tokens/s；decode 为基线 10.41–10.47、优化版 9.77–9.96 tokens/s。优化版整题仍需 36.94–37.29 分钟，峰值 MLX 内存由 52.32 降至 32.10GB。每题仅一组基线先行冷对照，尚无稳定提速、HTTP/MTP/缓存组合或广泛质量验收。[200K 对照归档](benchmarks/fused-tasks-200k-v1/manifest.json)。

融合注意力数值对照已完成：完整 204842-token 跨文件题，在相同基线续写前缀下比较 32 个位置，最高概率 token 全部一致，平均 KL 为 0.0006813；三个短任务共 96 个位置也全部一致。这是固定前缀数值诊断，不是独立生成的答案质量或提速验收。后续真实任务配对与有限 HTTP 组合测试见上方，候选保持默认关闭。[200K 复核归档](benchmarks/fused-distributions-200k-v1/manifest.json)。

完整 204800-token 分阶段诊断已复核：100 个连续输入块、四个窗口共 512 条模块计时，首 token 与三次基线一致。在 196608-token 位置处理 2048 tokens，全注意力合计 33.31 秒，占已计时模块约 77.5%；MLP 7.10 秒，线性注意力 2.56 秒。该比例仅适用于该同步诊断窗口，不是整次请求的耗时占比，也不代表提速或质量验收。[诊断归档](benchmarks/prefill-windows-200k-v1/manifest.json)。

200K 注意力分组优化的三组 AB/BA/AB 对照已完成：每次完整输入 204800 tokens、固定生成 64 tokens，三组 token 全部一致。首 token 提速比中位数为 0.999609，未观察到整体提速；峰值 MLX 内存从 52.31GB 降至 37.36–39.63GB。保留为默认关闭的内存优化候选，不视为语义质量或 HTTP 缓存验收通过。[三组完整归档](benchmarks/head-group-full-model-200k-repeated-v1/manifest.json)。

27B 双 200K 资料切换已完成独立复核：8/8 回答与裸 JSON 格式正确，四组完整历史、正文、reasoning 和输出长度一致；实际输入 204807–204940 tokens，四次热请求均从 RAM 复用原资料长度减一的完整前缀。切回 A/B 的首个可见回答为 1.99 / 2.07 秒、完整回答为 9.10 / 8.99 秒；带历史重复提问为 14.53 / 17.26 秒出现正文、21.35 / 24.11 秒完成，对应禁用 SSD 新进程冷对照为 2398.62 / 2428.60 秒。热 decode 10.41–10.84 tokens/s，首次冷请求仍需 40.28–41.38 分钟。配置为非 MTP 缓存归一化、仅末尾检查点、48GB RAM 缓存及 96GB SSD 上限，观测 MLX 峰值 85.94GB。本轮热请求均为 RAM 命中；后续独立的新进程测试已验证两份资料的 SSD 恢复（见下方）；两份合成资料和重复提问也不代表广泛日常质量或稳定性能。[完整归档](benchmarks/document-switching-27b-200k-v1/manifest.json)。

27B 的独立 SSD 恢复验证也已完成：A/B 分别在新进程、空 RAM 缓存下恢复 204806 / 204838-token 前缀，两次磁盘命中均有计数证据；输入完整、答案及输出长度与历史冷请求一致。首个可见回答为 6.14 / 5.45 秒，完整回答为 12.86 / 12.20 秒，decode 为 11.01 / 10.97 tokens/s，MLX 峰值约 56.65GB；模型启动另需 6.09 / 4.06 秒，不计入请求耗时。这是两次恢复观测，不代表稳定 SSD 延迟或冷 prefill 加速。[恢复归档](benchmarks/document-switching-27b-restart-v1/manifest.json)。

Flash 的 RAM＋SSD 双 200K 流程已完成独立复核：8/8 回答及裸 JSON 格式正确，4 组完整历史与输出一致，实际发生 2 次 SSD 恢复。A/B 首次请求为 565.24 / 588.79 秒；切回为 24.56 秒（SSD）/ 6.51 秒（RAM）；追问为 10.58 秒（RAM）/ 24.79 秒（SSD），对应禁用 SSD 的新进程冷对照为 554.43 / 563.11 秒。实际输入 204807–204940 tokens，观测 MLX 峰值 103.39GB，配置为 32GB RAM 缓存、48GB SSD 上限。详见[完整归档](benchmarks/document-switching-disk-200k-v1/manifest.json)。这是两份资料、每请求一次的有界测试，不代表多文档稳定性能或广泛日常质量；首次冷处理仍需约 9–10 分钟，27B 相同八请求流程也已完成，见上方；其热请求均命中 RAM。

双 200K 切换的内存诊断已定位到保存 B 最终快照时：所需余量 15.925GB，可用 15.340GB，差 0.585GB，触发整份 A（约 5.955GB）淘汰；32GB 缓存容量上限并未用满。两份冷回答均正确，A 回访全量重算后主动停止。[诊断归档](benchmarks/document-switching-admission-200k-v1/manifest.json)。RAM＋SSD 后备测试已完成上述有界复核；冷对照使用禁用磁盘恢复的新进程，因为后端清缓存接口只清 RAM。该诊断不作为计时基准或切换验收。

两种高频 200K 场景均为必测项：同文档连续追问，以及不同长文档切换。最新切换筛查使用两份完整 204807 / 204839-token 资料。原配置下 27B 回访 A 开始全量重算，Flash 完成 A 后缓存已被淘汰；Flash 改为仅保留末尾检查点后，两份冷回答均正确，prefill 为 389.39 / 394.41 tokens/s、decode 为 13.44 / 13.72 tokens/s，但处理 B 时仍淘汰 A，回访重算，故未通过切换验收。已停止失败回访并[归档部分证据](benchmarks/document-switching-final-only-flash-v1/manifest.json)。这不是完整八请求测试，该轮未运行 27B 的末尾检查点候选；当前 RAM＋SSD 配置已完成八响应及四组冷／热复核，热请求均命中 RAM；后续内存淘汰诊断及 Flash RAM＋SSD 复核已完成，见上方。

长上下文注意力分组候选已完成两层筛查：独立 SDPA 的 48 次观测均逐元素一致，200K KV 单组计算的观测峰值从 21.49GB 降至 6.37GB；这是随机激活微基准，不能换算为整模型内存或速度。完整 27B 的 8K/32K 四组配对均保持 64 个生成 tokens 一致，32K 峰值从 23.24GB 降至 21.27GB，但正式首 token 耗时分别增加约 7.7%/15.7%，并观察到后台 CPU 活动。因此不作为通用加速开关启用。[整模型筛查归档](benchmarks/head-group-full-model-screen-v1/manifest.json)保留负载限制。完整 204800-token 单组对照现已完成：仅在 KV 达到 51200 后启用分组，64 个生成 tokens 一致；预填充 87.13→90.82 tokens/s，首 token 2350.52→2255.04 秒，峰值 MLX 内存 52.31→37.36GB，decode 11.09→10.13 tokens/s。单组观测且基线期间有 CPU 准备工作，不能认定稳定提速；合成输入输出一致性不等于语义质量通过。[200K 归档](benchmarks/head-group-full-model-200k-v1/manifest.json)。服务默认路径未改变。

最新同任务对比：三类跨文件推理分别使用约 6K 与实际 204818–204842 tokens 输入，27B MTP、27B 非 MTP 缓存优化、Flash 缓存优化各做冷/热配对，共 36 次响应。内容 36/36 正确，裸 JSON 格式 34/36 通过（Flash 的一道 200K 题冷/热均加了代码围栏）；18 组正文、思考内容和输出长度一致，热请求均命中输入长度减一。200K 实测如下，单位为 tokens/s：

| 配置 | 冷 prefill | 冷 decode | 热 decode | 累计 MLX 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 27B + MTP | 85.8–89.4 | 11.4–11.7 | 11.4–11.6 | 69.1 GB |
| 27B 缓存优化、无 MTP | 89.1–89.6 | 10.7–11.0 | 10.8–10.9 | 68.8 GB |
| Flash 缓存优化 | 384.1–400.2 | 11.0–13.3 | 13.4–14.0 | 104.4 GB |

Flash 首次整题约 9.8–10.6 分钟，27B 两种配置约 39.1–40.7 分钟；热请求分别约 53–96 秒、57–101 秒，输出长度随模型和任务变化。decode 包含思考 tokens，模型加载不计入请求耗时。每个条件仅一次观测；三个合成任务不代表广泛日常质量，配置间有多项差异，不作 MTP 单项因果提速结论。[完整归档](benchmarks/diverse-reasoning-v1/manifest.json)保留原始输入、响应和独立审查；[统一 workload 汇总](benchmarks/workload-summary-v52.json)保留此前失败与未完成项。

新增完整仓库行为测试：两题均输入超过 206K tokens，三种配置共 12 次冷/热响应，内容 12/12 正确，六组正文及思考内容完全一致；但 12/12 带代码围栏，裸 JSON 格式全部失败。27B MTP 冷 prefill 83.34–83.44 tokens/s、decode 10.29–11.02 tokens/s；27B normalized 分别为 82.91–86.58、10.41–10.66；Flash-Next 分别为 283.28–283.94、12.95–13.44。Flash 冷请求总耗时 14.5–18.2 分钟、热请求 137.7–352.7 秒；27B 冷请求 41.5–47.8 分钟、热请求 106.3–371.7 秒。decode 包含思考，热请求完整复用除最后一个 token 外的输入；推理长短影响用户等待时间。同一仓库两题、每条件一次观测，不代表稳定性能或日常验收。[独立复核与原始响应](benchmarks/repository-behavior-200k-v1/manifest.json)。

27B 最大上下文诊断新增结果：同一份完整 261265-token 历史，Q4 在原生与 VLM 后端均误取相邻记录；本地 Q8 转换在原生后端冷请求中正确输出 `97804441`。Q8 总耗时 3411.93 秒，输入准备约 76.62 tokens/s，MLX 峰值 75.17GB；只生成 9 tokens，不作为稳定 decode 基准。输入、评分与运行源码哈希已复核。[Q8 归档](benchmarks/native-27b-8bit-261k-v1/manifest.json)。Q4/Q8 来自不同社区转换且内存/缓存预算不同，不能单独归因于量化；单题通过也不代表最大上下文广泛质量通过。统一结果已更新至 [v31](benchmarks/workload-summary-v52.json)，保留失败与未验收项。

27B 热缓存优化已完成三次基线与三次候选的交错重复对照（AB/BA/AB）：每组均为完整 206849-token 冷请求及精确热回放。12 次响应全部正确，正文、reasoning 和 699 个输出 tokens 全部一致。热 decode 中位数从 3.20 提升至 10.06 tokens/s（候选范围 9.51–10.88），热请求总耗时中位数从 220.25 降至 71.54 秒，约 3.08×；冷 prefill 中位数 82.43 / 81.38 tokens/s，未显示提升。两种配置进程累计 MLX 峰值中位数均为 69.34GB。详见[重复测试归档](benchmarks/27b-warm-normalize-repeated-v1/manifest.json)。两者均未使用 MTP，仅切换缓存归一化；这是同一道代码题的重复观测，不代表广泛质量、P95 或并发可靠性，仍保持非默认。

27B 四槽缓存已通过顺序交错验证：完整 204801-token 请求后插入两个独立短请求，再重复原请求，命中 204800 tokens；冷请求 2313.05 秒，重复请求首字 1.97 秒、总耗时 3.03 秒，四个答案均正确，长请求正文、reasoning 与输出 token 数一致，MLX 峰值 68.85GB。相同 24GB 缓存预算的短输入对照中，两槽丢失原前缀、四槽保留。[200K 归档](benchmarks/27b-cache-interleaving-200k-v1/manifest.json)及[短对照归档](benchmarks/27b-cache-interleaving-short-v1/manifest.json)。这是一次顺序交错检索测试，不代表并发吞吐、稳定性能或广泛质量；未改变默认配置。


27B 单请求热缓存优化的专用端口重测已通过有界完整会话复核：四组短会话冷/热一致，十轮 204801–205227-token 检索全部正确，九次完整前缀复用；末轮清缓存后正文、reasoning、输入及输出 token 数与热请求一致。首轮冷请求 2406.65 秒，末轮冷回放 2420.13 秒，九次热追问正文首字中位数 3.61 秒，观测峰值 MLX 68.85GB。日志中 19 次请求均与测试顺序一致、在途数为 1。[完整归档](benchmarks/27b-warm-normalize-conversation-v2/manifest.json)。这是四个重复追问的合成检索测试，不代表广泛任务质量或稳定加速倍数，候选仍为可选；两槽/四槽短请求插入对照及四槽 200K 缓存保留验证已完成，见下方。

27B 热缓存优化的首次长会话验证未通过完整流程：首轮 204801-token 检索正确，但期间服务收到两个非测试请求；它们完成后，完整缓存从约 13.58GB 降至 0.37GB，第一次追问开始全量重算。已主动停止该未完成追问并[保留部分失败证据](benchmarks/27b-warm-normalize-conversation-interference-v1/manifest.json)，不作单请求速度结论，也不归因于归一化补丁本身。相同的四轮短会话及 200K 十轮／末轮冷回放已改用 18083 专用端口并完成验证，见上方。四槽顺序插入两个短请求的保留验证已通过；并发请求与更多任务切换仍需验证。

27B 单请求热缓存候选已通过短代码及 206849-token 跨文件推理配对复核：200K 基线/候选各一次冷、热请求，四次答案均正确，完整正文、reasoning 与 699 个输出 tokens 一致。热请求从 221.49 秒降至 69.74 秒，decode 从 3.18 提至 10.32 tokens/s，本次总耗时约 3.18×；冷请求为 2546.26 / 2527.43 秒，未解决首次长输入等待。四次服务观测峰值 MLX 均为 69.34GB。[200K 复核归档](benchmarks/27b-warm-normalize-200k-v1/manifest.json)保留完整证据。这是单次有序配对，尚不代表稳定加速或广泛质量保证；候选仍为非默认，完整 200K 十轮历史及末轮冷回放验证已完成，见上方。

Query-sliced prefill 完整模型筛查已完成：8K 与 32K 各一组预热、三组正式配对，所有配对的 64 个输出 token 一致；正式首 token 延迟速度比中位数为 1.0017× / 0.9886×，未显示稳定的实际加速，因此保持禁用。详见[完整记录与复核](benchmarks/query-sliced-prefill-full-model-v1/manifest.json)。这不是 200K 候选验证，也不替代语义质量评测。27B 单请求热缓存优化的短任务和 200K 代码配对已完成，结果见上方。

最新进展：27B 原生服务完成 200K 会话分支对照，7 次请求全部答对，4 组冷/热一致性通过独立复核。实际输入 204801–204844 tokens，三个冷请求耗时 2265.38 / 2280.03 / 2297.52 秒，四个热请求 1.59 / 2.39 / 1.64 / 2.46 秒；均输出 12 tokens 并正常结束。使用 4 个完整前缀槽、60GB 缓存预算、110GB MLX 预算，无 MTP，观测峰值 MLX 83.45GB。[归档](benchmarks/native-27b-followup-branches-200k-v1/manifest.json)保留原始证据。Flash-Next 已完成相同检索 fixture，但后端与缓存策略不同，不将耗时比作为受控跨模型加速倍数。单一检索问题的分支通过尚不足以证明多样化任务的日常可用性。完整模型 query-sliced prefill 筛查已完成，未见稳定收益，仍保持禁用。

面向 Apple Silicon 的 **Qwen 文本推理优化与多模型 workload 评测工具**。当前已实测 Qwen3.8-27B 与 Qwen3.8-Flash-Next。包含可复现基准、prefill 参数扫描、完整混合状态前缀缓存、可选 greedy prompt lookup，以及本地聊天 API。加速结论按场景分别验证，不宣称所有请求都快于 MLX-LM。

</details>

## 目标与顺序

最终验收扩展为 **Qwen3.8-27B 与 Qwen3.8-Flash-Next 的多模型、分 workload 对照**，执行范围见 [workload 计划](benchmarks/workload-plan.json)。覆盖短结构化任务、200K 检索、跨文件推理、真实源码阅读、长会话和生成速度。 已有证据统一索引见 [分 workload 汇总](benchmarks/workload-summary-v52.json)，包含实际缓存命中、失败及尚未完成的验收项。跨模型比较质量/延迟/内存取舍；每个模型内部另做固定权重的优化对照。两种模型均已有本机短任务与 200K 推理记录；当前结果见文末，仍未完成日常使用验收。

`scripts/audit_workload_tokens.py --models /path/to/27b /path/to/flash-next --suites suite.jsonl --output audit.json` 使用每个模型的本地 tokenizer 和聊天模板审计同一套输入，记录文件哈希、实际 tokens、输出预算与剩余历史空间，保持原始任务不变。默认以 204,800 输入 tokens 检查长上下文门槛；短任务未达到此门槛是正常的。计数审计不代表推理或质量通过。当前两份固定版本 tokenizer 对已有短任务、200K 检索、代码与源码样本的计数一致。

1. **上下文预算与评测**：按实际聊天模板计算输入和预留输出，建立质量及 HTTP 端到端测试。
2. **完整会话缓存与内存预算**：同时保存 Attention KV、卷积和循环状态，验证输出历史复用与采样一致性。
3. **8K–256K 分档验证**：测量冷处理、缓存追问、正确性和内存，结合任务质量确定日常推荐范围。
4. **原生 MTP**：在同一目标权重上交错对照，记录输出一致性、接受率、验证成本和端到端净收益，再验证服务集成。
5. **冷 prefill 优化**：根据 profiling 筛选内核或执行策略，完成数值与完整模型回归后才调整默认路径。

性能验收：在同一模型、量化、输入及硬件上，相对于固定版本原生 MLX-LM，重复测得净收益。量化变化必须单独评估质量，不能混进同精度加速数字。单个微基准结果不代表模型整体提速。不预设无法验证的倍数目标。

上述五项的第一轮工程验收已完成，历史证据见 [验收记录](benchmarks/acceptance.json)。**当前目标已提高到实际输入至少 204,800 tokens（200K）的日常使用**，输出、思考和对话增长另留预算，服务保留原生 262,144 上限。此前 8K–32K 仅为已测的小规模基线，不是最终验收目标；256K 曾出现检索错误。200K 的加速后端质量、多轮完整缓存、冷/热对照和速度优化正在补测，未完成前不宣称达标。

200K 首轮实测发现上游 VLM 默认 600 秒 token 等待期限会在冷 prefill 中途取消请求。启动器现提供 `--token-timeout`，默认 7200 秒，并显式传入 `MLX_VLM_TOKEN_QUEUE_TIMEOUT`；客户端也需要足够长的读取超时。这是长输入兼容修复，不是速度提升。测试器保留 SSE 服务端错误，超时不得计为答案质量失败或成功完成。

[200K MTP 第二轮记录](benchmarks/200k-mtp-cache-diagnosis-v2/)中，204,801-token 冷首字为 2,281.90 秒，首题及两次追问正确，追问首字为 3.34/3.08 秒；第三次追问因缓存驱逐开始完整重算，已停止诊断，不能宣称十轮会话通过。可选 `--release-finished-mtp` 清理完成批次持有的请求缓存和暂停生成器，保留独立 APC 快照；该补丁的短任务输出及十轮冷/热一致性已通过；尚未默认启用，长上下文验证进展如下。

[200K MTP 第三轮会话记录](benchmarks/200k-mtp-conversation-v3/)已完成这一合成检索会话：十轮答案通过，九次完整前缀复用通过，实际输入由 204,801 增至 205,227 tokens。最后一轮清空缓存后重放，正文与思考内容一致，服务报告 cached tokens 为 0。该轮冷首字 2,265.26 秒，热追问首字 3.55 秒；整个会话首次冷首字 2,513.82 秒。四个不同问题在十轮中重复，不能视为十个独立质量样本。下载与测试有重叠，冷/热数字只是本次观测；更多任务质量、受控速度对照与 Flash-Next 仍待完成，当前尚不认定 200K 日常使用达标。

[200K 跨文件推理记录](benchmarks/200k-mtp-code-v3/)中，206,849-token 合成代码任务的冷/热答案均正确，正文与思考文本一致，热请求复用 206,848 tokens。冷请求整题 2,354.57 秒；热请求 1.59 秒开始思考、60.44 秒出现正文、61.89 秒完成，共生成 699 tokens。缓存消除了大部分重复 prefill，仍需承担思考与正文生成时间。这个单任务结果不能外推为复杂任务普遍通过或所有热请求都能三秒回答；非 MTP 同提示基线、真实源码任务与 Flash-Next 比较仍在进行。

[200K 真实源码的 4096 输出预算记录](benchmarks/200k-mtp-repository-budget4096-v3/)未通过：206,635-token 输入完整处理后，生成 4,096 tokens 仍全部为思考，正文为空，以 `finish_reason: "length"` 结束；整题已耗时 2,652.72 秒。模型在浮点数换算上反复推敲，不能把这次运行当作任务完成。该失败与此前通过的合成检索/代码题分开保留；[8,192 输出预算诊断](benchmarks/200k-mtp-repository-budget8192-v1/)也未通过：热请求复用 206,634 tokens，耗时 709.70 秒，最终正文仍为空并以 length 结束。[推荐采样配置诊断](benchmarks/200k-mtp-repository-sampling-v1/)中，思考模式以 5,262 输出 tokens 给出正确的八字段 JSON，热请求耗时 469.28 秒、正文首字 460.71 秒；非思考模式 13.34 秒完成，却把 `memory_max_bytes` 错算为 25,769,803,776（应为 24,000,000,000）。两者均复用 206,634 tokens，各仅一次观测；不能以关闭思考的速度替代质量验收。[关闭 MTP 的 4,096 输出预算对照](benchmarks/200k-repository-greedy4096-baseline-comparison-v1/)也以 length 结束且正文为空，4,096 tokens 的思考文本与 MTP 逐字一致；本例的思考循环在不使用 MTP 时同样出现。非 MTP 冷 prefill 为 2,211.87 秒，整题 2,592.92 秒；两次运行下载重叠条件不同，不能据此宣称受控加速倍数。后续热请求和推荐采样对照见下文。

[Flash-Next 外部 PLE 准备记录](benchmarks/flash-next-external-ple-preparation-v1/)已通过 CPU 校验：22 个完整权重分片、3,767 个张量，非 PLE 权重由硬链接保留，全部 PLE 数据通过原始文件范围映射访问，未裁剪专家或 PLE。此前 912 次 PLE 行抽样与直接解量化结果一致。后续实际加载及有界 200K 测试已完成，结果见文末；连续缓存驻留仍有失败。

[8,192 输出预算的非 MTP 热请求](benchmarks/200k-repository-greedy8192-baseline-comparison-v1/)已结束：耗时 2,541.62 秒，decode 为 3.23 tokens/s，复用 206,634 tokens，仍以 length 结束且正文为空；全部思考文本与对应 MTP 热请求逐字一致。该诊断包含 3 秒调用栈采样，不能作为受控提速倍数。

[推荐采样的非 MTP 对照](benchmarks/200k-repository-sampling-baseline-comparison-v1/)也已完成：思考模式 1,630.06 秒、decode 3.23 tokens/s，八字段 JSON 正确；MTP 对应观测为 469.28 秒、11.26 tokens/s。两者均复用 206,634 tokens，5,262 输出 tokens 的正文和思考文本逐字一致。非思考模式的错误也逐字一致：非 MTP 34.99 秒、MTP 13.34 秒，都把内存字节数算错。每个配置仅一次观测，尚未做交替重复，不能当作受控提速倍数或普遍质量保证；即使较快的思考结果也仍需约 7.8 分钟。

[非 MTP 的 205K 完整会话末轮对照](benchmarks/200k-baseline-conversation-v1/)冷/热回答均为正确的 `60454861`，正文与推理一致，也与 MTP 冷回放一致。冷请求 2,187.78 秒、正文首字 2,187.05 秒；同输入热回放 4.16 秒、正文首字 1.78 秒，复用 205,226 tokens。两次各输出 9 tokens，decode 分别为 11.00 和 3.38 tokens/s。仅一对末轮观测，不等同非 MTP 十轮全验收；MTP 原末轮暖请求的未缓存后缀不同，不将其直接作为同条件 MTP 提速倍数。

[206,849-token 跨文件代码的非 MTP/MTP 冷热对照](benchmarks/200k-code-baseline-comparison-v1/)四次均正常结束，JSON 答案正确，699 输出 tokens 的正文与推理文本逐字一致。非 MTP 冷/热总耗时 2,273.21 / 210.60 秒，MTP 为 2,354.57 / 61.89 秒；两组热请求均复用 206,848 tokens。非 MTP 热 decode 3.34 tokens/s，MTP 11.58 tokens/s，说明暖缓存路径仍需进一步优化。每组仅一对观测，未交替重复，不能据此宣称稳定提速倍数；冷 prefill 仍是主要等待来源。

非 MTP 的 200K 热请求诊断发现：完整前缀命中后，decode 仍约 3–4 tokens/s；源码及 CPU 复现显示单条批量缓存每步提取/合并会丢失预留容量。实验开关 `--normalize-single-warm-cache` 在后缀 prefill 前只转换一次，限定 27B、非 MTP、单请求且未量化 KV，默认关闭。64 层真实缓存类的 CPU 状态检查通过，原前缀、KV 偏移和循环状态数值/类型保留；6144-token 与 206849-token 完整模型冷/热代码配对均已通过，四次输出分别一致；连续 200K 检索会话及末轮冷回放已通过；四槽顺序交错短请求已通过 200K 保留验证；并发、重复性能与更广泛质量仍需验证，尚不能认定日常使用通过。

## 当前设备与模型

- 开发设备：M3 Max，40 核 GPU，128 GB 统一内存。
- 基础模型：[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。初始运行权重：[MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)。
- 本机快照：`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`。配置使用 `qwen3_5` 架构名称，不应将配置名误认为模型版本。
- 配置：64 层、hidden size 5120，48 层 linear attention + 16 层 full attention，Dense FFN。优化方向不包括 Strata 的 MoE 专家缓存。
- 第一阶段只测文本，保留权重和 tokenizer 原样，不启用远程模型代码。

## 运行

需要 Apple Silicon、可访问 Metal GPU 的会话，以及 Python 3.11+。

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e .

# 默认只查找本地 Hugging Face 缓存，不下载模型
qwen-booster bench --lengths 512 --steps 2048 --tokens 32 --repeats 1 --output results/smoke.jsonl

# 分块调优，包含原生默认 2048 基线
qwen-booster bench --lengths 512 2048 8192 32768 --steps 512 1024 2048 4096 --tokens 128 --repeats 3

# 也可以显式指定本地模型目录
qwen-booster bench --model /path/to/Qwen3.8-27B-4bit --output results/custom.jsonl

python -m pytest -q

# 真模型缓存一致性：中文、代码、重复恢复、前缀扩展与分叉
qwen-booster verify-cache

# 实验性 prompt lookup：检查输出、接受率、回放成本和净速度
qwen-booster verify-lookup

# 单独运行，避免与 benchmark 同时争用 GPU
qwen-booster profile --length 512

# 根据 MLP 热点做投影合并实验（当前未发现收益，不用于服务）
qwen-booster verify-fusion
```

`results/*.jsonl` 保存元数据和每次测量，旁边的 `*.summary.json` 保存中位数与候选配置。为保护原始数据，工具拒绝覆盖已有报告。

## 测量口径

- 模型加载时间单独记录，计时前权重已 materialize；每次生成创建全新缓存。
- 输入是精确长度的重复中英代码 token 序列，用于负载比较，不是质量评估集，也不是真实对话吞吐的代表。
- 使用 MLX-LM 原生 `generate_step`、greedy、固定生成数量，忽略 EOS；输入 tokenization 和输出 detokenization 不计时。
- TTFT 从提交生成到首 token 返回；`prefill_tps = 输入长度 / TTFT`，包含首 token 计算和采样，并非纯矩阵算子吞吐。
- `decode_tps = (输出数量 - 1) / (最后 token 时间 - 首 token 时间)`，保留原生异步流水行为。
- 每种配置先预热，重复实验随机交错。记录随机种子、配置、版本、输入摘要与输出 token IDs。
- 内存是 MLX 峰值分配量（十进制 GB），不是系统总驻留内存。
- 分块推荐目前是本轮数据中的最低中位 TTFT，标记为 provisional。正式默认值需独立复测和正确性验证。
- Metal 在某些沙箱/无头会话中不可见，需要在可访问 GPU 的本机会话运行。

## 实验路径

`runtime.Engine` 提供单模型、单请求的精确前缀缓存。缓存有 LRU 和字节预算，保存所有层的 KV、卷积和循环状态；只有完整匹配的快照可以复用，不对循环状态做任意位置截断。预算计算快照的逻辑字节数，不是全进程内存上限。MLX 0.32.3 的数组 deepcopy 共享底层数据；活跃请求追加、工作区和分配器仍需额外预算，应结合峰值测量判断。

`lookup.lookup_generate` 是默认关闭的 greedy prompt lookup 原型。拒绝草稿后恢复整个快照，再回放已接受输入；连续三次零接受会关闭本次请求的草稿。该回放可能抵消推测收益，因此需要真实模型一致性和速度测试，不能以草稿接受率代替净加速。

`profile` 在各 block 边界同步 GPU，帮助定位候选热点。同步会改变执行行为；报告不包含 embedding/head 等开销，不能将 block 耗时总和直接当成真实端到端延迟。

## 本地 API

```sh
qwen-booster serve --port 8080
# 仅在需要尝试代码复制类任务的推测生成时显式开启
qwen-booster serve --port 8080 --lookup

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"请解释二分查找"}],"max_tokens":128,"temperature":0,"stream":true}'
```

服务只绑定 localhost，每次执行一个请求。支持文本 `messages`、`max_tokens`、`stream`、`enable_thinking`，以及原生路径的 `temperature`、`top_p`、`top_k`、`min_p`、`presence_penalty`、`repetition_penalty`、`seed`。默认仍是温度零、thinking 和 lookup 关闭；原生路径已通过 4 组真实模型采样一致性测试（thinking 开关、贪心惩罚、固定种子的非零采样），这不等同于广泛任务质量评估。lookup 仅支持无惩罚的贪心生成。暂不支持 tools、图像和完整 OpenAI API 参数集，不支持的字段会明确报错。

`preserve_thinking` 和 `reasoning_effort` 传入模型自带模板；assistant 历史可附带 `reasoning_content`。思考输出通过 `reasoning_content` 返回，正文通过 `content` 返回。客户端须保留并传回推理内容以复用完整历史。采样惩罚使用完整输入与生成历史，不因缓存命中丢掉输入前缀。

提供 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/cache/stats`，以及 POST `/v1/cache/reset`。响应包含缓存和 MLX 内存遥测；流式响应的最后一个 JSON 事件携带 `timings`，其 `choices` 为空。清缓存只清除本服务的内存前缀快照。

`--context` 默认 32768（包含输入与预留输出），`--cache-mb` 默认 2000，`--step` 默认 2048。缓存预算不能覆盖的长前缀正常计算但不保存。断开连接会关闭生成 iterator；当前 GPU 操作和 prefill 不能即时中断。该接口用于本机实验，不是多人并发服务。

`--memory-mb` 默认 64000，设置 MLX 分配指导值并在 prefill 分块和生成步之间检查活跃分配；这不是操作系统硬内存上限，也无法中断单个正在执行的 GPU 操作。`--allocator-cache-mb` 默认 2000，限制空闲分配器缓存。prefill 每块完成后清理空闲缓存。首次初筛在一个 64K 用例完成后因系统 swap 明显增长而停止；加入上述预算和清理后，v3 已完成到 256K 的会话请求，8K–128K 检索检查通过，256K 有一次答错。单次诊断通过仍不能证明该档位的普遍任务质量。

服务启动时检查 `--context` 不超过模型配置的原生上限；请求超限直接报错，不截断输入。
`/health` 返回配置上限、原生上限以及尚未完成质量验证的标识。配置为 262144 不代表已经通过 256K 验收。
原生生成正常完成后保存包含输出的完整混合状态；仅当所有 KV 层的消费位置与 token 前缀一致时提交。
取消请求不提交生成尾部，lookup 实验路径仍只复用输入前缀。

长上下文诊断（当前进行中）：

```bash
python scripts/build_context_suite.py --output results/context-suite.jsonl
qwen-booster serve --context 262144 --cache-mb 24000 --port 18080
python scripts/evaluate_api.py --suite results/context-suite.jsonl \
  --output results/context-quality.jsonl --endpoint http://127.0.0.1:18080/v1 \
  --label '记录后端版本、权重版本、量化与缓存条件'
```

测试集覆盖 8K、16K、32K、64K、128K、256K，每档三个事实位置，按实际模板 token 数为输出预留 128 token。
这是可复现的合成检索/跨记录关联诊断，不能替代真实代码任务、thinking 模式、量化质量对照及多轮会话评估。
HTTP 脚本保存原始回答、完整流标识和首正文延迟；SSE 块数不冒充 token 数。
含内联 `<think>` 的响应暂不提供首正文指标。重复运行可能命中缓存，比较时必须明确缓存条件。

完整会话测试每档只冷载入一次，再追加三次不同位置的记录查询：

```bash
python scripts/build_context_suite.py --headroom-tokens 1024 --output results/conversation-context-suite.jsonl
python scripts/evaluate_long_conversation.py --suite results/conversation-context-suite.jsonl \
  --output results/context-conversations.jsonl --endpoint http://127.0.0.1:18080/v1
python scripts/summarize_conversations.py results/context-conversations.jsonl \
  --output results/context-conversations-summary.json
```

v3 完整记录见 [会话测试归档](benchmarks/long-context-native-v3/)。每档 1 次冷请求与 3 次追问；24 次请求均完成，23 次检索检查通过。8K–128K 全部通过，256K 最后一次答错相邻记录。数据来自同一服务，`--cache-mb 24000 --memory-mb 64000 --allocator-cache-mb 2000`，默认 prefill 分块 2048。

| 档位 | 初始实际输入 tokens | 冷请求首正文秒 | 追问首正文秒范围 |
| --- | ---: | ---: | ---: |
| 8K | 7,168 | 32.49 | 0.45–0.48 |
| 16K | 15,359 | 76.35 | 0.52–0.54 |
| 32K | 31,744 | 168.38 | 0.56–0.59 |
| 64K | 64,512 | 405.84 | 0.75–0.80 |
| 128K | 130,048 | 1,114.10 | 1.07–1.12 |
| 256K | 261,119 | 3,409.86 | 1.84–1.88（其中一次答错） |

冷请求与追问的问题不同，这张表不构成成对加速比；单会话的小样本不报告 P95。输入预留了追问与输出空间。该结果验证的是合成记录检索和前缀复用，尚未确定日常推荐范围。

256K 末次查询要求记录 `005721` 的项目编号 `97804441`，模型返回相邻记录 `005722` 的 `30158402`。输入中的目标记录唯一，响应完整且命中 261,227 tokens 的前缀缓存。该档位的 MLX 观测峰值约 61.72 GB；能够执行且缓存追问较快，不等同于质量通过。错误归因尚待冷路径或其他精度对照，不能直接归因于缓存或量化。

质量补充与候选优化入口（以下质量初筛已运行，内核实验仍待验证）：

```bash
python scripts/build_task_suite.py --output results/task-quality-suite.jsonl
python scripts/build_long_task_suite.py --output results/long-task-quality-suite.jsonl
python scripts/evaluate_api.py --suite results/task-quality-suite.jsonl \
  --output results/task-quality-native.jsonl --endpoint http://127.0.0.1:18080/v1 \
  --label 'native MLX-LM 0.32.0, fixed 4-bit snapshot'
# 再对 long-task-quality-suite.jsonl 执行相同评测；8-bit 参考另启服务、记录独立标签。

# 停止其他 GPU 推理后执行；这些实验不修改服务默认路径。
python scripts/benchmark_sdpa.py --output results/sdpa-screen.jsonl
python scripts/benchmark_fused_prefill.py --output results/fused-prefill-screen.jsonl
python scripts/benchmark_prefill_gemm.py --output results/prefill-gemm-screen.json
```

MTP 使用独立环境的 `requirements-vlm.lock`，目标权重保持原来的 4-bit 快照，另加载匹配的 MTP head。`scripts/benchmark_mtp.py` 做同后端开关对照，`scripts/native_mtp_reference.py` 生成跨后端 token 参考；`qwen_booster.vlm_server` 是独立的实验服务启动器，原有 MLX-LM 服务仍为默认路径。

MTP 基准支持 `--memory-mb` 和 `--allocator-cache-mb`，默认分别为 64000/2000；较大模型需根据实际加载与上下文峰值选择预算。工具核对目标模型和草稿类型，并记录模型标识、预算与 PLE 存储配置。Flash-Next 必须使用 `qwen4_exp_mtp` 草稿，不能复用 27B 的草稿头；其速度成绩仍待实测。

[MTP 复测归档](benchmarks/mtp-kernel-confirm-v1/)：三个短输入任务、每次生成 128 token、每组预热后交替复测三次。非思考模式比较块大小 2/3/4，三个任务均以 3 最快，配对端到端加速中位数分别为 **1.82×、1.63×、1.37×**；思考模式块大小 3 为 **1.38×、1.32×、1.31×**。全部 36 个正式对照对的同后端 greedy token 一致。这里测量的是短输入固定长度生成，不能外推为完整任务或长输入速度；VLM 与 MLX-LM 两个后端的基线输出并非全部相同，切换后端仍需独立任务质量和 API 验收。

冷 prefill 的强制融合注意力候选已完成初筛：[算子记录](benchmarks/sdpa-fused-screen-v1/)在长 KV 下约快 1.25–1.40×，但存在数值差异；[完整模型记录](benchmarks/fused-prefill-screen-v1/)中 2K 三次正式对照输出一致、TTFT 中位收益仅约 1%，8K 三次均改变 greedy 输出。因此不启用该候选；输出差异本身不等于任务质量退化，但不满足当前输出一致性验收条件。

[8K 分块 profiling 与 MLP 算子记录](benchmarks/prefill-profile-gemm-v1/)中，MLP 占同步诊断所测 prefill 模块耗时约 65.3%（同步会扰动执行，不等于无插桩性能占比）。实际权重展开后的矩阵乘在 512/2048 token 上约快 9–11%，观测数值差异为零；单 token 更慢且有差异，不能用于 decode。[正式实现交错复测](benchmarks/expanded-prefill-runtime-confirm-v1/)中，2K/8K 各三次正式对照输出 token 全部一致，TTFT 加速中位数分别为 **1.103×/1.082×**；额外常驻约 **34.2 GB**，准备耗时单独记录。计时两种模式都保留展开权重，公平比较计算路径；准备时间不计入 TTFT。提供默认关闭的 `--expanded-prefill` 选项。这是内存换取小幅 prefill 收益，不能把该比率与 MTP 比率直接相乘。


[MTP API 验收](benchmarks/mtp-api-quality-v1/)已确认真实草稿统计、流式/非流式完整响应，以及输入加预留输出超限时返回 400。关闭 APC 的同测试集对照中，greedy 八项的正文和推理文本全部一致，基线和 MTP 都是非思考 3/4、思考 4/4；固定采样也都是 7/8。各项仅一次 HTTP 观测，完整任务质量覆盖有限。`--cache-mb` 可显式开启内存 APC（默认 0，磁盘缓存关闭），短会话和约 32K 历史的缓存组合已完成下述验收。

[MTP 长任务验收](benchmarks/mtp-long-quality-v1/)的 8K/32K 思考用例均通过，非思考均失败，与原生服务的检查结果一致。该测试为有限的合成跨文件语义任务，开启 APC 且缓存状态混合，不用于冷启动速度比较。内存监测覆盖 32K 会话和长任务期间，系统 swap 未增长；RSS 不代表全部 GPU 分配。

展开权重的组合接入曾发现一项类型分派回归：[诊断记录](benchmarks/prefill-dispatch-investigation-v1/)。通用层包装隐藏了 `QuantizedLinear` 类型，使上游 MTP 精确验证退回不同计算路径，三项短任务推理文本发生变化。现已保留量化层类型及平坦参数布局，直接内核回归通过；相同 APC 配置的八项短任务正文和推理文本重新全部一致。[修复版服务验收](benchmarks/expanded-mtp-api-v2/)的四个 8K/32K 长任务未新增检查失败，十轮短会话逐轮与冷计算一致；[原生 API 验收](benchmarks/expanded-native-api-v1/)的 8K 思考题冷/暖回答和推理文本一致。正式实现速度复测已完成，[组合配置的约 32K 十轮会话](benchmarks/expanded-mtp-32k-v1/)也已完成，所有追问复用检查通过；仅末轮另做冷计算，答案一致，首字 0.783 秒对 159.982 秒。该配置为 MTP block 3、展开 prefill、4GB APC、80GB 内存指导值。

实验服务使用独立环境，避免替换原生服务依赖。先准备上述固定版本的目标目录，以及 `mlx-community/Qwen3.8-27B-MTP-4bit` 的 `b643c01b6d3b094e325edb6ebd832e16c486c575` 版本目录（包含 config、权重和 index）；启动过程只读取本地文件：

```sh
uv venv .cache/vlm-env --python 3.12
uv pip install --python .cache/vlm-env/bin/python -r requirements-vlm.lock
PYTHONPATH=src .cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model /path/to/Qwen3.8-27B-4bit --draft /path/to/Qwen3.8-27B-MTP-4bit \
  --block 3 --port 18081 --context 32768 --memory-mb 64000 \
  --allocator-cache-mb 2000
```

Flash 当前启动命令还依赖 external-PLE 存储视图。原始目标为 `mlx-community/Qwen3.8-Flash-Next-4bit`，本次固定版本 `07b5dc6c54600a359b87f1e53e7adf6351c72a2c`；完整下载目录与新建视图须位于同一文件系统。使用上述锁定的 VLM 环境准备本地视图：

```sh
.cache/vlm-env/bin/python - <<'PY'
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
mx.set_default_device(mx.cpu)
prepare_external_ple_model(
    Path('.cache/models/qwen3.8-flash-next-4bit'),
    Path('.cache/models/qwen3.8-flash-next-4bit-external-ple'),
    cache_rows=0,
)
PY
```

准备函数要求目标目录为空，已有本机视图无需重建。非 PLE 权重使用硬链接，PLE 通过 `ple-store.json` 引用原始目录；运行时必须保留原始权重。此步骤不裁剪专家或 PLE，也不代表模型推理验证通过。[本机准备审计](benchmarks/flash-next-external-ple-preparation-v1/flash-next-external-ple-prepare-v1.json)记录了完整性检查。MTP 草稿另需包含完整的 54 个张量，当前本机路径为 `.cache/qwen3.8-flash-next-mtp-4bit`；[草稿准备审计](benchmarks/flash-next-mtp-prepare-audit.json)核对了去除命名前缀前后的所有张量数据。

客户端使用 `http://127.0.0.1:18081/v1`，请求的 `model` 填启动时目标模型的完整路径。传入 `enable_thinking: true` 可启用思考，流式结果分别返回正文与 `reasoning_content`；后续轮次应保留完整历史。`stream_options: {"include_usage": true}` 返回用量和 MTP 统计。需要缓存时加 `--cache-mb 24000`；[十轮短会话对照](benchmarks/mtp-apc-short-v1/)已逐轮通过 warm/cold 文本一致性检查，[约 32K 十轮会话](benchmarks/mtp-apc-32k-v1/)也已完成：首次首字 166.48 秒，后续缓存首字约 0.72–1.07 秒；仅最后一轮另做完整冷计算，回答一致，首字 0.75 秒对 180.32 秒。这是一次配对观测，前九轮不能宣称通过冷计算一致性对照。原生服务的 `qwen3.8-27b` 别名不适用于这个独立服务。

质量初筛见 [4-bit 任务记录](benchmarks/task-quality-native-v1/)、[匹配贪心参数记录](benchmarks/task-quality-greedy-native-v1/) 和 [4/8-bit 对照](benchmarks/quality-4bit-8bit-v1/)。所有跨精度比较都核对了测试集 SHA；两种精度使用相同 tokenizer/template。每项只运行一个固定种子，属于有限诊断。

| 测试 | 4-bit 非思考 / 思考 | 8-bit 非思考 / 思考 |
| --- | --- | --- |
| 四个短任务，模式各自采样设置 | 3/4 / 4/4 | 2/4 / 4/4 |
| 四个短任务，同为贪心且无惩罚 | 3/4 / 4/4 | 3/4 / 4/4 |
| 同一跨文件语义题，8K 与 32K 输入档位 | 0/2 / 2/2 | 0/2 / 2/2 |

非思考计费题在两种精度中均出现计算或零贡献输入判断错误；不能把这些错误直接归因于 4-bit。思考配置在上述检查中都通过，但这不是全面编程能力或量化无损的证明。思考与非思考长输入的输出预留不同，实际输入长度分别记录在测试集中。

[32K 十轮会话记录](benchmarks/session-32k-native-v1/) 验证了连续前缀复用，最后一轮与清空缓存后的完整历史重算答案相同；仅最后一轮做了冷对照。该轮首正文约 0.416 秒 / 181.264 秒，一次成对观测，不报告重复实验的加速倍数。

**第一轮小规模基线为 8K–32K，复杂任务保留思考模式。当前最低目标为 200K 实际输入，尚在补测。** 以下是历史基线的复现配置。对这台 128 GB 设备，可用 `qwen-booster serve --context 32768 --cache-mb 24000 --memory-mb 64000 --allocator-cache-mb 2000` 复现会话测试的预算设置；请求显式传 `enable_thinking: true`，客户端保留完整历史和 `reasoning_content`。默认 2000 MB 缓存不足以保存接近 32K 的完整状态，需按用途增加。

复杂任务可先使用本轮长任务通过的配置：`enable_thinking: true`、`reasoning_effort: "medium"`、`max_tokens: 2048`、`temperature: 1.0`、`top_p: 0.95`、`top_k: 20`、`presence_penalty: 0`；复测时另设 `seed: 1729`。`max_tokens` 同时覆盖思考和正文，不能沿用简短示例的 128 或原生默认 256 来判断复杂题能力；`finish_reason: "length"` 表示预算耗尽，不能当成回答完成。32768 上下文预留 2048 输出后，模板化输入最多为 30720 token。缓存节省输入处理时间，思考与正文的生成时间仍然存在。

### 这台 128GB Mac 的历史小规模复现配置

- 需要维持原生 MLX-LM 路径：使用上面的原生服务与 24GB 会话缓存。
- 希望提高生成速度：使用独立 VLM 服务，加 `--draft /path/to/Qwen3.8-27B-MTP-4bit --block 3 --cache-mb 24000`，上下文保持 32768。已完成相同后端的 MTP 开关对照，但 VLM 与 MLX-LM 的输出不保证逐 token 相同。
- 长输入首次处理较频繁、能接受约 34.2GB 额外常驻内存：在 VLM 启动命令中使用 `--expanded-prefill --cache-mb 4000 --memory-mb 80000`。这组配置完成了 32K 会话验收；展开权重默认关闭，速度证据限于本机所测 2K/8K 原生 prefill，对更长输入不承诺相同比率。

```sh
# 已准备独立环境和固定版本的本地权重；同时开启 MTP、缓存与可选 prefill 优化。
PYTHONPATH=src .cache/vlm-env/bin/python -m qwen_booster.vlm_server \
  --model /path/to/Qwen3.8-27B-4bit --draft /path/to/Qwen3.8-27B-MTP-4bit \
  --block 3 --context 32768 --port 18081 --cache-mb 4000 \
  --memory-mb 80000 --allocator-cache-mb 2000 --expanded-prefill
```

以上都是第一轮单用户文本配置，不代表当前 200K 目标已验收。长上下文、复杂编程、多种随机种子和实际项目任务的质量覆盖仍有限，不能把这些检查称为普遍“不降智”认证。第一轮性能实验与服务接入已完成，256K 质量失败、额外内存代价和适用范围均保留在结果中。

64K–128K 仅在材料确实需要时扩容，并接受当前数分钟至约 19 分钟的首次处理成本；这些档位已做检索测试，复杂任务覆盖不足。256K 已验证能完成请求，但出现检索错误且首次处理约 57 分钟，暂不推荐为日常质量配置。超出服务预算时明确报错，不自动截断或摘要输入。MTP 与展开 prefill 的收益仅使用各自已验证范围内的数据，不外推到 256K。

## 初始实验结论（MTP 接入前）

参考环境：本机 M3 Max / 128 GB，固定的 Qwen3.8-27B MLX 4-bit 快照，MLX 0.32.3 / MLX-LM 0.32.0。归档见 [原始测量](benchmarks/m3-max-qwen3.8-27b-4bit/)。所有比率只在同一实验内比较。

| 实验 | 观察 | 决定 |
| --- | --- | --- |
| 完整前缀状态复用 | 6 个真模型案例与原生 token 一致；两个短提示的引擎 TTFT 从 358–513 ms 降至约 98 ms | 服务默认启用有界前缀缓存；这些延迟不含 tokenization/HTTP |
| Prompt lookup：代码复制 | 128 个固定输出 token；预热后交错三次，中位耗时对应 1.505× 加速，三次均接受 64/70 个草稿 token | 提供显式 `--lookup`；结论仅限本轮复制案例 |
| Prompt lookup：普通中文 | 输出一致，但只有 0.955×，三次均接受 9/15 个草稿 token | 不默认启用推测生成 |
| 2K prefill 分块 512 vs 2048 | 首轮约 1.148×，独立复测仅 1.022×；复测输出完全一致；峰值内存从 18.21 GB 降至 16.52 GB | 速度证据不足，保留默认 2048；512 可作为省内存选项 |
| 合并 MLP gate/up 投影 | 算子数值误差为 0，生成 token 一致；修正权重生命周期后的复测约为原生 prefill 0.978×、decode 0.987× | 保持实验代码，不接入服务 |
| GDN Metal 布局 | 原生 packed 与 unpacked 在测试维度逐位一致；原生 packed 在测试的所有长度更快 | 保留上游默认布局 |
| 32K 长输入原生基线 | 预热后三次中位数：TTFT 201.7 s、prefill 162.4 tok/s、decode 约 15.9 tok/s、峰值内存 23.24 GB | 作为后续长输入优化对照，不是加速成绩 |

首轮矩阵存在明显运行状态漂移，不能用它选择新的速度默认值。短输入独立实验与长时间连续负载的结果也有明显差异。这组初始实验没有证明首次 prefill 或通用 decode 普遍加速，也没有用新的量化精度替换基线。当时尚未接入原生 MTP；后续 MTP 证据见上文。项目未实现全新的 Metal 算子、图像推理或 llama.cpp/GGUF 对照。

复现实验脚本：`scripts/benchmark_gdn.py` 对比上游内核；`scripts/smoke_api.py` 启动并关闭临时本地服务，测试两种生成模式；`scripts/archive_results.py` 归档这一组固定命名的结果。不要同时运行多个 GPU 性能实验。

## 工程改造：git、vLLM 风格缓存与开源测试集（2026-10-07）

- **版本管理**：源码、测试、脚本和 `benchmarks/` 证据已纳入 git；首次提交为 `14305e5`。`results/` 与 `.cache/` 继续忽略，避免把 294GB 模型和可重生成原始输出放进仓库。每个归档内的 `src/` 冻结快照仍是不可变证据。
- **vLLM 风格混合缓存**（默认关闭）：`qwen_booster.hybrid_cache` 用块哈希链（`hash(parent, block)`）替代逐 token 前缀扫描，只保存块对齐检查点；循环状态从不截断，命中后最多重放一个块加保留的末 token。`HybridEngine` 通过 `serve --hybrid-cache --hybrid-block-size 256` 启用；磁盘层 `qwen_booster.apfs` 用 APFS `clonefile` 做写时复制备份、safetensors 临时文件加原子改名写入，`serve --cache-disk-dir <dir> --cache-disk-mb <MB>` 启用，`/v1/cache/reset` 会同时清磁盘。旧 `PrefixStore` 仍是默认路径。
- **混合缓存真机验证已完成有界检查**：M3 Max 128GB、Qwen3.8-27B MLX 4-bit 快照 `10c35caa…`、8193-token 合成提示、greedy 8 tokens。冷 prefill 39.1 秒；同提示热命中边界 8192、prepare 0.001 秒、输出逐 token 一致；仅末 token 变化和追加 8 token 的提示同样命中 8192；无复用冷对照输出一致；同进程磁盘恢复 0.006 秒、新进程恢复 0.004 秒，输出一致。HTTP 端到端（6020 tokens）：冷请求 27.5 秒、热请求命中 5888 并重放 132 tokens、总 prepare 0.83 秒、正文一致。归档见 [hybrid-cache-native-v1](benchmarks/hybrid-cache-native-v1/manifest.json)。这只验证缓存机制，**不是吞吐、质量或日常使用验收**，也没有覆盖 200K 档位。
- **观测与结果查看（Langfuse 评估）**：本项目为本地优先、单请求实验服务，验收依赖逐字节 JSONL/manifest 与冷/热对照；而 Langfuse v4 自托管需要 Web+Worker、PostgreSQL、Redis、ClickHouse、S3 五件套，云端版会把提示与回答传出本机，且它不采集 MLX 峰值内存、缓存准入、块命中这类决定验收的字段。因此**暂不引入 Langfuse**。替代方案是新增零依赖离线报告器 `scripts/render_report.py <results.jsonl> --output report.html`，把测量 JSONL 渲染成带数值条的静态 HTML；原始 JSONL 仍是唯一事实来源。若后续服务变为多用户/多并发，再考虑 OpenTelemetry（上游 mlx-vlm 目前没有 tracing 钩子）或自托管 Langfuse。
- **开源测试集**：`grading.py` 新增 `choice`、`numeric`、`contains_all`、`contains_any`、`regex` 判定；`scripts/build_opensource_suite.py` 离线转换本地 RULER 风格 NIAH、LongBench、GSM8K、MMLU 导出为现有 suite 格式（`evaluate_api.py` 直接可用），不下载、不改写原始任务，未通过的数字字段会跳过并计数。**转换成功不等于质量验收**：开源集存在污染、模板与语言差异，长上下文档位仍需按本机 tokenizer 审计实际输入。
- 新增测试总数：244（原 200 + 缓存与归档 28 + 评分、转换、报告与并发 16）。

## 内网服务评估：chat + 并发 + 运维诊断 agent（2026-10-07）

目标场景从"本机单请求实验"变为"内网多用户 chat / 运维诊断 agent"，因此对服务底座做了能力排查与真实并发测量。结论：**可行，但要改掉实验默认值，并接受专用硬件约束**。

**能力现状（上游 pinned mlx-vlm 服务端）**：已内置连续批处理与背压（`--max-num-seqs`，映射 `MLX_VLM_MAX_NUM_SEQS`）、`--api-key`、`/health` 与 `/metrics`（Prometheus）、tools/function calling（本机加载 `qwen3_coder` 工具解析器）、结构化输出、OpenAI 与 Anthropic 兼容端点。此前项目启动器 `qwen_booster.vlm_server` 把它们硬编码关掉了（`--max-num-seqs 1`、绑定 `127.0.0.1`）；现已增加 `--max-num-seqs`、`--host`、`--api-key` 透传，默认仍是单序列 + localhost，按需开启。native mlx-lm 服务是单线程实验接口，不适合并发 chat。

**Flash-Next 并发实测**（M3 Max 128GB，4-bit external-PLE，16K context，APC 关闭，112-token 中文诊断题，greedy 64 tokens，热态第二轮，服务端 `--max-num-seqs 4`）：

| 并发 | 单路 decode tokens/s | 聚合 tokens/s | 首 token | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 24.0 | 24.0 | 0.66s | 热态单路 |
| 2 | 19.5–21.0 | 28.7 | 1.1–1.5s | |
| 4 | 14.8–15.1 | 40.2 | ~2.0s | 批处理满编，聚合约为单路的 1.7 倍 |
| 8 | 12.2–14.0 | 36.8 | 2.1s / 第二波 8.9s | 超过 cap 的请求排队，吞吐平台化 |

同层内所有请求 greedy 文本完全一致（批处理不失真）；首个请求有约 25 秒冷启动/编译开销，服务启动后应先预热。**MTP 草稿（block 3）在本提示上反而更慢**：单路 14.0 tps、4 路每路 5.5 tps，尽管输出与基线逐 token 一致；这与项目早期其它短任务 1.3–1.8× 的证据冲突，说明 MTP 收益高度任务相关，**并发服务默认不要开**，按负载复测后再定。归档见 [flash-concurrency-v1](benchmarks/flash-concurrency-v1/manifest.json)，仅为单提示两轮的有界观测，不是容量规划。

**架构建议**：内网网关（认证、限流、审计、排队）→ `vlm_server --host <内网地址> --api-key <token> --max-num-seqs 2–4`（OpenAI/Anthropic 任选）；交互 chat 用 Flash（更快 prefill 与 decode），27B 更慢但质量证据更好，适合异步/批处理任务。**Flash 峰值约 105GB、27B 约 85GB，128GB 单机无法同时常驻，一台一模型或按需切换**。长文档诊断不要把 200K 冷 prefill 当交互（Flash 也要 9–10 分钟）：用检索/截断策略、会话钉住和共享前缀预热；长会话并发时按 context 预算下调 `max-num-seqs`。项目自研的单行缓存优化（borrow/credit/recheck）在批处理路径不生效，并发下走上游 APC，尚未做并发命中验收。观测先沿用 `/metrics` + JSONL + `render_report.py`；用户变多后再考虑在独立机器自托管 Langfuse/OTel，理由见上节。

## 参考

- [Strata](https://github.com/Niko1221/Strata)：分阶段优化、状态复用、推测解码和数值回归的方法。
- [MLX-LM](https://github.com/ml-explore/mlx-lm)：当前参考后端；项目没有把原生能力宣称为自研加速。
- [MLX Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)：后续热点算子扩展路径。


Flash-Next 已实际加载完整模型的 external-PLE 存储视图并完成[基础 API 与四轮缓存检查](benchmarks/flash-next-cache-screen-v1/)：输入 7,722–7,829 tokens，各轮冷/热正文和推理一致，后三轮复用前缀，热请求 0.52–0.77 秒。首轮混合了代码题与项目约束，不能当作语义质量通过。[相同八项短任务](benchmarks/flash-next-short-quality-v1/)为 6/8 正确：思考模式 4/4，非思考模式 2/4，后者在跨文件语义和约束更新上答错。以上均为单次有界检查；[实际 204,801-token 测试](benchmarks/flash-next-200k-cache-failure-v1/)首轮检索正确，总耗时 505.84 秒、正文首字 505.06 秒，但随后因完整前缀缓存未保留而停止：`exact_stores=0`、`memory_skips=3`、`exact_resident_bytes=0`。因此十轮与末轮冷回放未完成，200K 多轮尚未验收；当前正在诊断缓存内存准入。

后续[内存准入诊断](benchmarks/flash-next-apc-admission-diagnosis-v1/)复现了失败：单份完整 200K 快照约 5.9–6.0 GB，小于 12 GB APC 容量，但默认额外预留与复制预算合计要求约 23.47 GB 空余，当时仅约 20.7–20.8 GB。[额外预留调整为 6 GB 的实验](benchmarks/flash-next-reserve6000-v1/)完成了正确的冷请求（476.96 秒）和第一次热请求（2.37 秒，复用 204,800 tokens），随后仍因内存准入失败清空全部快照，十轮及最终冷回放未完成。该实验峰值 MLX 内存约 103.33 GB；诊断日志开销和单次观测不构成受控性能比较。默认参数未修改，下一项实验是保留一份完整检查点，验证多轮驻留；不削减上下文、专家或 PLE。

[单份检查点实验](benchmarks/flash-next-single-checkpoint-v1/)仍在首次热请求后清空缓存：两轮检索均正确，热请求总耗时 2.09 秒，MLX 进程峰值约 97.43 GB。热请求的新快照需约 17.93 GB 余量，而当时可用约 16.58 GB，未通过原有准入检查。下一轮保留单份完整检查点，将额外预留由 6 GB 调为 4 GB，原内存准入检查继续生效；该参数尚未验收，也未成为默认值。

相同八项短任务已做[双模型独立重新评分](benchmarks/multi-model-short-quality-comparison-v1/)：27B 为 7/8（思考 4/4、非思考 3/4），Flash-Next 为 6/8（思考 4/4、非思考 2/4）。这是四类题各测试思考开关的单次观察，不能据此认定整体模型优劣。两批缓存和上下文配置不同，报告保留每题耗时及 token 遥测，但不计算受控加速倍数。

Flash-Next 的[200K 十轮检索与末轮冷回放](benchmarks/flash-next-200k-conversation-v1/)已通过有界检查：一份完整 APC 检查点、4 GB 额外预留下，十轮全部答对、九次后续请求复用前缀、每轮结束仍有驻留快照。首轮正文首字 478.27 秒，九次热请求首字中位数 1.69 秒；最终 205,227-token 完整历史冷回放 `cache_n=0`，首字 490.95 秒，与热请求正文及推理一致。MLX 进程峰值约 98.52 GB。全模型和上下文未裁剪，默认配置未改变；这是四种重复追问的合成检索单次检查，仍需长代码、真实仓库、分支会话及重复测量，不等同 200K 日常使用验收。

同一 206,849-token 跨文件代码题的[三配置冷/热对照](benchmarks/multi-model-200k-code-comparison-v1/)已完成独立评分，六次均正确且正常结束；各配置内部冷/热正文和推理一致，冷请求缓存为零、热请求复用 206,848 tokens。总耗时如下，都是单次观测，模型之间输出长度不同，不能作为受控加速倍数或整体优劣结论。

| 配置 | 冷请求总耗时 | 热请求总耗时 | 输出 tokens（含思考） |
| --- | ---: | ---: | ---: |
| 27B 基线 | 2273.21 秒 | 210.60 秒 | 699 |
| 27B MTP | 2354.57 秒 | 61.89 秒 | 699 |
| Flash-Next 基线 | 535.11 秒 | 55.57 秒 | 770 |

Flash-Next 本轮热请求正文首字 53.63 秒、生成约 14.12 tokens/s，MLX 进程累计峰值 97.60 GB；生成思考仍占主要等待时间。真实仓库的四配置结果见下文。

同一完整真实仓库问题的[四配置双模型对照](benchmarks/multi-model-200k-repository-comparison-v1/)已结束：Flash-Next 为 3/4，27B 基线为 1/4。两者 4096-token 贪心均耗尽预算；Flash-Next 在 8192-token 贪心下用 6091 tokens 正确完成，而 27B 仍未完成；思考采样两者正确；非思考采样仅 Flash-Next 八字段全对。这是一个问题的四种配置，不代表四个独立任务。Flash-Next 思考采样因缓存被驱逐重新处理完整输入，耗时 1251.51 秒；非思考采样复用 206634 tokens，耗时 9.75 秒。缓存状态不同，不能用总耗时计算受控模型加速倍数；复杂任务间缓存持续驻留仍待修复与验证。

[连续相同 200K 请求诊断](benchmarks/flash-next-warm-reserve-diagnosis-v1/)复现了缓存丢失：三次 204,801-token 请求答案完全一致且正常结束，但缓存 token 数依次为 0、204800、0，总耗时 549.63、2.11、495.73 秒。恢复后约 5.95 GB 逻辑缓存仍对应约 11.92 GB 预留，快照在状态合并时已被驱逐。新增默认关闭的 `--credit-live-warm-cache` 候选，仅用于 Flash-Next 无草稿单请求路径，保留准入检查及至少一份状态复制/增长预留；107 项 CPU 测试通过；下述 GPU 回归显示相同请求通过、追加历史仍失败，尚未认定完整修复有效。

默认关闭的预留修正在[三次相同 200K 请求回归](benchmarks/flash-next-warm-reserve-credit-200k-v1/)中通过：缓存 tokens 为 0、204800、204800，总耗时 488.67、1.74、1.70 秒，三次均正确、正常结束，正文和推理与原策略一致；原策略第三次曾回到 495.73 秒冷计算。该修正只改变已恢复状态的预留计量，不裁剪输入或模型；单次诊断观测不代表受控性能倍数。四轮短上下文冷/热回归也通过；随后完整历史增长回归失败，默认保持关闭。

[预留修正的追加历史回归](benchmarks/flash-next-credit-append-failure-v1/)在两轮后停止：204,801-token 冷请求与 204,851-token 追问答案均正确，追问复用 204,800 tokens、正文首字 2.11 秒，但响应结束后快照驻留为零。保存新快照要求约 15.93 GB 余量，驱逐旧快照后只剩约 12.38 GB，因此准入被拒绝。十轮、最终冷回放及后续长代码回归均未完成；不能用相同请求的通过结果替代追加历史验收。下一步需测量行提取与快照复制的实际内存重叠，再验证有界的复制优化。

[快照分阶段内存诊断](benchmarks/flash-next-snapshot-allocation-v1/)再次复现追加失败：冷请求行提取的 MLX 活跃内存增量为 0，快照克隆增加约 5.96 GB；追加请求行提取却先增加约 5.03 GB，之后准入失败，尚未执行克隆。该观测支持优先优化单行提取的重复物化；它不证明所有数组的独占所有权，也不是速度基准。随后已实现默认关闭的 `--borrow-single-checkpoint-row` 候选，111 项 CPU 测试通过。

[借用单行视图的短回归](benchmarks/flash-next-snapshot-views-short-v1/)已通过四轮冷/热正文及推理一致性检查，输出也与前一候选一致。该候选仅在 Flash-Next 无草稿的完整单行缓存布局中延迟行提取复制，保留原管理器内存准入和独立快照克隆。CPU 检查覆盖 QSA、GDN、PLE 状态及底层数组修改隔离；200K 十轮回归正在运行，尚未证明长会话修复。

[借用行视图的 200K 十轮回归](benchmarks/flash-next-snapshot-views-200k-v1/)已通过独立复核：十轮答案正确、九次前缀复用、每轮完整快照驻留，最后 205,227-token 冷回放缓存命中为零，正文及推理与末轮热请求一致。热追问正文首字中位数 1.72 秒；末轮冷回放正文首字 566.71 秒。这是四种追问重复的合成检索单次检查，复制优化继续默认关闭；长代码生成与更广质量回归尚待完成。

[复制优化的长代码回归](benchmarks/flash-next-snapshot-views-long-code-v1/)已通过：同一 206,849-token 思考代码任务连续三次均正确、正常结束，均生成 770 tokens，正文和推理与原基线一致；缓存命中为 0、206848、206848，完成时间为 578.14、57.55、58.43 秒。该结果表明本次长生成后第三次仍可复用前缀；它是单个任务的顺序观测，不是受控加速倍数或全面质量证明。候选继续默认关闭，原生最大上下文、分支会话、Flash-Next MTP 和更多 workload 尚待验证。

实验性磁盘 APC 可通过 `--disk-cache-dir <本地目录> --disk-cache-mb <十进制MB>` 显式开启，同时需要正数 `--cache-mb`。默认关闭；启用后会在指定目录持久化完整提示缓存。小型 CPU 检查已验证 Flash-Next 的 QSA/GDN/PLE 状态同步落盘、关闭后重新打开索引、恢复结果与原状态一致；这尚不证明 200K/原生上限大快照的内存开销、恢复速度或推理输出。当前最大上下文测试仍使用纯内存 APC，未切换配置。

[Flash-Next 原生上限实测](benchmarks/flash-next-native-boundary-v1/)完成三位置、六次请求：实际输入 262,016 / 262,016 / 262,015 tokens，预留 128-token 输出预算；六次均正确且正常结束，每组正文与推理一致。前部冷/热耗时 708.79 / 2.84 秒，重复命中 262,015 tokens；中部 662.12 / 697.76 秒、尾部 685.81 / 701.23 秒，后两组重复命中均为零，因内存准入失败重新计算。此结果证明本轮合成任务能运行到原生容量边界，但三组只有一组复用成功，最大长度的稳定低延迟未验收。实际输出仅为短答案，不能外推为完整 128-token 输出、广泛质量或多轮增长测试。

[完整 Flash-Next 磁盘重启恢复短测](benchmarks/flash-next-disk-short-v1/)已通过：8,063-token 输入在 1 MB 内存 APC 预算下落盘，服务完全退出并重启后恢复，正文及推理一致、答案正确、`disk_hits > 0` 且没有驻留内存快照。快照约 345.55 MB，首次请求 24.64 秒、重启后请求 7.31 秒。耗时包括本轮恢复和生成，不能外推 200K；204,801-token 同类恢复测试正在运行，磁盘选项仍默认关闭。

[204,801-token 磁盘重启恢复](benchmarks/flash-next-disk-200k-v1/)通过单用例检查：首次请求 539.26 秒，写入约 5.96 GB 完整快照；进程退出后重新加载模型，磁盘恢复请求 15.04 秒，复用 204,800 tokens，答案、推理及输入 token 数一致。服务统计确认磁盘命中且内存快照驻留为零。该耗时从请求开始计算，不含模型启动时间；这是短答案检索的单次观测，原生最大长度、长代码、分支及多轮磁盘恢复尚未验证。

分支缓存检查脚本 `scripts/evaluate_cache_branches.py` 支持修改末条用户消息及 `--branch-mode followup` 两种模式；后者保留真实基础回答和已输出推理，再分别生成追问。两条分支均与独立冷回放比较正文、推理、输出长度和正确性，并检查返回原分支后的复用。该 CLI 仅用于内存 APC：启动前拒绝启用磁盘缓存的服务，因为 `/cache/reset` 不会删除磁盘快照，磁盘分支的冷对照需要新存储目录及服务重启。可通过 `--backend native|vlm` 分别使用 27B 与 Flash-Next 服务的实际遥测字段，并检查冷／热输入和输出 token 数一致。143 项 CPU 测试通过，分支 GPU 验证尚未执行。

[27B 原生上限错误样例冷回放](benchmarks/native-27b-failure-replay-v1/)已复现原错误：完整 261,265-token 历史、缓存命中为零、正常结束，耗时 3455.59 秒，仍回答 `30158402`，正确答案为 `97804441`。因此该错误无需缓存复用也会出现；这尚不能区分模型、量化与后端原因，也不证明历史缓存完全正确。同提示重复请求已主动停止：24 GB 缓存预算在保留约 17.28 GB 生成结束快照时驱逐输入快照，不能形成计划中的热对照。该次运行按不完整配对归档，不将主动终止记录为模型崩溃；后续原生分支及重复测试需按快照容量重新配置预算。

[Flash-Next 200K 追问分支检查](benchmarks/flash-next-followup-branches-200k-v1/)通过独立复核：实际输入 204,801–204,844 tokens，7 次回答均正确且正常结束，四组冷／热正文、推理和输入输出 token 数一致。两个分支热请求耗时 2.33 / 2.29 秒，对应零缓存冷回放 509.97 / 530.28 秒；返回原对话 1.79 秒。首次相同请求只复用 202,752 tokens，耗时 7.92 秒；后续复用 204,800 tokens。此配置使用三个完整检查点、24 GB 内存缓存、110 GB MLX 预算，MLX 累计峰值约 104.36 GB，区别于此前单检查点配置，优化继续默认关闭。这是同一检索问题的两个追问分支；修改历史用户消息、长生成、重复测量、27B 分支对照及更广泛质量仍待验证。
