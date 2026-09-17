# 显存预算闸门：设计、机制与保真度边界

这是整个任务族的核心部件。它要回答的问题是：

> **verifier 怎么判断「这份推理代码能在 8GB 显存的设备上跑起来」？**

结论先行：**不是在物理 8GB 卡上跑，也不是让 agent 自己上报，而是「硬闸门 + 外部测量」两条独立机制叠加。**

---

## 1. 为什么不能只靠物理卡

Harbor 的 GPU 白名单是 `T4 / L4 / A10 / L40S / A100-40GB / A100-80GB / H100 / H200 / B200`，
**没有 8GB 这一档**（`check-gpu-types.sh`）。所以「在 8GB 卡上跑」这件事必须被**模拟**。

三种可选做法，我选第三种：

| 做法 | 问题 |
|---|---|
| 声明一张最小卡（T4 16GB）然后不管 | 16GB 依然是预算的两倍，没改的代码照样能跑过 |
| 用 MIG 切一小块 | H100 的 MIG profile 是 1g.10gb / 2g.20gb 等，**没有 8GB**；且 harbor 的 schema 不暴露 MIG 配置 |
| **硬闸门 + 外部测量** | 见下 |

---

## 2. 两条独立机制

### 机制 A：硬闸门（enforcement）

```python
torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes, 0)
```

PyTorch 的 caching allocator 会被限制在预算内。**超了就是真的 `torch.cuda.OutOfMemoryError`**，
和真实 8GB 卡上的行为完全一致——不是「报个数字然后放过」。

在 8GB 卡上预算是 8GiB，换算比例 > 1，会被 `min(1.0, ...)` 夹到 1.0，等于不额外限制——
此时物理卡本身就是约束。

### 机制 B：伪装设备显存报告（fidelity）

**这条容易被忽略，但缺了它整个测量就失去意义。**

如果只在 H100 上量峰值，agent 的代码会看到 `total_memory = 80GB`，于是走「大卡分支」
（更大的 tile、更少的切分、直接 materialize）。那么量出来的峰值描述的是**另一个分支**，
和「在 8GB 卡上会怎样」无关。

所以 harness 在导入 agent 代码**之前**把这些 API 都改掉：

| 被改的 API | 改成 |
|---|---|
| `torch.cuda.get_device_properties(0).total_memory` | 预算值 |
| `torch.cuda.mem_get_info()` | `(预算 - 已用, 预算)` |
| `torch.cuda.memory_stats()` 里的 `*.total_bytes` / `*.pool_bytes` | 预算值 |

这样 agent 的 `if torch.cuda.get_device_properties(0).total_memory < 9 * 2**30:` 这类
自适应逻辑会**和真实 8GB 卡走进同一个分支**。

### 机制 C：进程外测量（ground truth）

前两条都在被测进程内部，理论上被测代码可以干扰（比如自己再调一次
`set_per_process_memory_fraction(1.0)` 把闸门拆掉）。

所以**打分用的数字不是它们报的**，而是父进程通过 NVML 独立采样的：

```python
nvmlDeviceGetComputeRunningProcesses(handle)   # 按进程取 usedGpuMemory
nvmlDeviceGetMemoryInfo(handle)                # 整流设备 used
```

父进程按 50ms 采样一次取最大值。被测进程无法影响这个数字。
最终 `peak = max(NVML 进程, NVML 设备, torch allocated, torch reserved)`——
取四个来源的最大值，任何单一来源被绕过都不影响结论。

---

## 3. 为什么光有闸门还不够：必须同时校验输出

「跑起来了、峰值 ≤ 8GiB」这件事可以被各种廉价手段满足：

- 直接 `return`，不计算
- 只算一部分
- 把输出缩小
- 用低精度近似到面目全非

所以 verifier 同时做**保真度比对**：用 grader 自己镜像里的 oracle 实现，在**同一问题规模**下
跑一遍，和 agent 的 `frames.u8` 逐像素比：

```
mean|diff| <= 2      max|diff| <= 12      PSNR >= 40 dB     (0-255 uint8)
```

再加一条 `test_pipeline_config_unmodified`，断言 `CFG` 里的 `seq / frames / out_hw / layers`
没被动过——**否则「把问题改小」就是最简单的通过方式**。

三条合起来才构成完整判定：
**闸门（不超预算）+ 跑通（不崩）+ 保真（算对了）+ 规模未缩水（题没被改小）**。

---

## 4. 保真度边界：这套模拟**不**等价于真实 8GB 卡

必须说清楚哪些差异是真实存在的，否则就是在包装结论。

| 差异 | 影响 | 本任务的处置 |
|---|---|---|
| **报告伪装只覆盖 torch API** | agent 若直接读 NVML / `nvidia-smi` / `/proc/driver/nvidia`，仍能看到真实卡（H100 80GB） | 未堵。属规格问题：题面已声明预算是 8GiB，**用真实卡容量做分支**不在规格内。若要堵，可在 verifier 镜像里把 `nvidia-smi` 换成返回预算的壳脚本——**尚未实现** |
| **caching allocator 只管 torch 分配** | 走 `ctypes`/cuBLAS 裸分配的显存不计入闸门 | 本任务两处热点都是 torch 算子，覆盖到了。若任务引入自定义 CUDA 扩展，就不成立 |
| **卡不同 → kernel 选择不同** | H100 与消费卡的 SM 数、可用 attention kernel、tile 自动调优都不同。在 H100 上「刚好 7.9 GiB」的方案，在 8GB 消费卡上未必成立 | **无法消除**。所以闸门设成 8 GiB 预算而不是「刚好卡满」；oracle 实测峰值 1.12 GiB，留了 7 倍余量 |
| **碎片化行为不同** | 真实卡上长期运行后的碎片会让可用显存低于标称 | 未建模。在 H100 上碎片程度不同 |
| **SM 代际差异（sm_90 vs sm_120）** | 任何**架构特定**的优化（Triton autotune、`sm_120` 专属 kernel）在 H100 上无法验证 | 这是**任务设计层面的约束**：本任务只考核「显存预算」，不考核 kernel 的绝对性能，所以可移植。见 `TASK_FAMILY.md` |

**一句话**：这套闸门可靠地回答「峰值显存有没有超过 8GiB」，
**不能**可靠地回答「在真实 8GB 消费卡上一定能跑、一定够快」。
把话说在这里，比让 reviewer 自己发现要好。

---

## 5. 已验证的事实（本机 RTX 5060 Laptop 8GB）

> 本机开发用的 python 是 `C:/Python314/python.exe`，**torch 2.11.0+cu128**。
> 容器里钉的是 **torch 2.14.0+cu130**（`environment/Dockerfile`），两者不要求一致——
> 保真度是按容差比较的，而且闸门机制只依赖 torch 的 allocator API，与版本无关。

`dev/smoke.py budget` 的三臂实验：

| 代码 | 结果 | 峰值 | 判定 |
|---|---|---|---|
| `oracle_pipeline.py` | OK，2.2s | **1.12 GiB** | 通过 ✅ |
| 只修注意力，保留整卷 decode | **OOM** | 10.35 GiB | 失败 ✅（证实「只修一处不够」） |
| 起始 `pipeline.py` | **OOM** | 9.32 GiB | 失败 ✅ |

跑完整 grader（8 个测试）时的实测峰值会略高：oracle **1.66 GiB**
（torch 分配 1.14 / reserved 1.66），starter **9.38 GiB 后 OOM**。
两者差异来自 smoke 的 fixture 与 grader 的 reference 构造方式不同，属正常。

起始版本 OOM 时请求的字节数是 `9,663,676,416` —— 正好等于 `8 heads × 24576² × 2 bytes`，
即自注意力分数矩阵的理论值。**数值可复现，不是巧合。**

注意这台机器只有 8GB，所以比例夹到 1.0，闸门实际由物理卡提供；
在 H100 上跑同一个实验，闸门由 `set_per_process_memory_fraction` 提供，
结论一致（起始版本一样在 9.66 GB 处 OOM）。

**NVML 在本机不可用**（开发 python 没装 `pynvml`），所以本地跑出来的是
`nvml process=0.00, nvml device=0.00`，峰值回退到 torch 的读数。
容器里 `nvidia-ml-py` 是装好的，进程外采样会真正生效。

---

## 6. 复用到别的任务时要改什么

`budget_harness.py` / `vram_sampler.py` / `vram_probe.py` 三个文件是**任务无关**的，
换个预算值就能复用。要改的只有：

1. `BUDGET_GIB`（本任务 8.0）
2. `test_outputs.py` 里的 `REQUIRED_CFG` —— 声明「不许缩水的规模量」
3. oracle 实现（每个任务各自的正确参考）
4. 保真度阈值 —— 取决于 agent 被允许引入多大数值偏差

第 4 点最需要判断：**阈值定太松，近似解能过；定太紧，合法优化过不了**。
本任务实测 oracle 与起始版本在小规模下**逐位一致**（max|diff| = 0），
所以 40 dB 的余量是充足的；若任务允许 fp8/int8 量化，这个阈值必须重定并经实验标定。

---

## 7. 三个只有跑「作弊臂」才会发现的缺陷

这三个都是我先跑通了「正确解通过」，然后跑「错误解」才暴露出来的。
**记在这里，因为它们是这个任务族里最容易重复犯的错。**

### 缺陷 1：显存峰值测试在「跑挂」时会误判为通过

nop 臂（未改动的起始代码）第一次跑出来是这样的：

```
test_pipeline_runs_within_budget   FAILED   ← 正确
test_peak_memory_within_budget     PASSED   ← 错误！
    peak memory OK: peak=7.86 GiB (nvml device=7.86, torch alloc=0.00)
```

**根因**：`set_per_process_memory_fraction` 让 torch 的分配**永远到不了** 8 GiB 以上，
所以「峰值是否超过预算」这个判据**在闸门正常工作时几乎不可能触发**。
更糟的是，当物理卡容量本身就 ≤ 预算（本机 8GB 卡、预算 8GiB）时，
NVML 读到的设备级峰值最多就是整张卡——它**物理上不可能超过预算**。

于是：一份 OOM 崩掉的代码，「峰值」是 7.86 GiB ≤ 8 GiB，测试通过。

**修法**：峰值测试必须加**运行成功**这个前置条件——
OOM 是预算的失败，不是「留在了预算内」的证据。

**通用教训**：`cap` 负责**拦截**，`measurement` 负责**抓越狱**，两者职责不能混淆。
峰值测试的真正价值是抓「绕过 cap 但跑完了」的情况
（agent 在自己进程里再调一次 `set_per_process_memory_fraction(1.0)`，
或走非 caching allocator 的裸分配）。这个价值只有在 `status == ok` 时才成立。

### 缺陷 2：输出没有依赖被考核的那段计算 ⭐ 更严重

`cheat` 臂（把增益系数从 `0.5` 改成 `0.2`，**显存完全合规但算错**）第一次跑出来是：

```
7 passed   ← 作弊通过了！
```

**根因**：我把 DiT 的输出用 `x.mean(dim=(0,1))` 折叠成增益。
`x` 是 24576 个 token 的表征，近似独立同分布，
**均值的标准差 ≈ 1/√24576 ≈ 0.006**，于是 `gain` 坍缩到 1.0 ± 0.1%。
注意力分支对最终像素的影响只有千分之几——**把整个 DiT 删掉也能过保真度检验**。

换句话说：显存闸门考核的是注意力那一大坨中间量，
但**输出根本不依赖它的结果**。任务在语义上是空的。

**修法**：把 readout 从「沿序列求均值」换成「**采样 token 位置 + 固定随机投影**」，
保持增益是 O(1) 且真正依赖每一个 token（通过 attention 全局耦合）。

**通用教训**（这条要写进每个任务的设计检查清单）：

> 保真度检验有意义的前提是：**被考核的计算必须真正决定输出**。
> 必须有一个夹具专门验证这件事——**把那段计算整个删掉，保真度必须挂**。

任何「把大张量坍缩成小标量再用于后续」的设计都有这个风险，
尤其是坍缩方式是**沿很长的轴求均值**的时候。

### 缺陷 3：参考实现退化成常量，保真度检验自我通过

修完缺陷 2 之后，两个作弊臂确实开始挂保真度了。但当时的模型是**数值不稳定**的：
权重用未归一化的 `randn` 初始化、每个 block 之间没有 LayerNorm，
激活值在 fp16 下溢出成 NaN，最终**参考实现和被测实现都输出全零帧**。

于是保真度比较报 `mean|diff| = 0`，**包括那个故意扰动计算的夹具也「通过」了**。
一个静默全常量的参考实现会把保真度闸门彻底关掉，而且不留任何痕迹。

**修法**（两处，缺一不可）：

1. 数值侧：权重按 `1/sqrt(fan_in)` 初始化 + 每个 block 加 `F.layer_norm`。
2. 判据侧：加一个**grader 自检** `test_reference_is_nondegenerate`，
   断言参考帧 `std > 5.0` 且子采样后不同取值 > 16。

**通用教训**：

> 只要判据是「A 与 B 是否接近」，就必须有一条断言**A 本身携带信号**。
> 否则 A = B = 常量 会让所有比较全过。
> 「两者一致」永远不等于「两者都对」。

这条自检还额外暴露了我本地驱动工具的一个 bug（漏调第 8 个测试），见
`HARBOR_TASK_FORMAT_CN.md` §10——**本地全绿但 CI 会多跑一个测试**。

---

## 8. 作弊臂清单（每个任务都必须建）

跑通正确解只证明了一半。这套夹具是用来证明**任务不是空的**。

实测结果（RTX 5060 Laptop 8GB，8 个测试，二进制 reward = 全过才 1.0）：

| 夹具 | 做了什么 | 通过 | reward | 峰值 | 失败点 |
|---|---|---|---|---|---|
| `oracle` | 正确且省显存的参考解 | 8/8 | **1.0** | 1.66 GiB | — |
| `starter` | 未改动的起始代码 | 4/8 | 0.0 | 9.38 GiB | OOM → 运行/峰值/产物/保真四项连挂 |
| `approx_wrong_output` | 显存合规但 gain `0.5→0.2` | 7/8 | 0.0 | 1.66 GiB | **仅保真度**（mean\|diff\|=10.99 > 2.0） |
| `skip_dit` | `return x` 删掉整个注意力栈 | 7/8 | 0.0 | 1.30 GiB | **仅保真度**（mean\|diff\|=13.76 > 2.0） |

注意后两行：**它们都成功骗过了显存闸门**（峰值 1.66 / 1.30 GiB，远低于 8 GiB），
唯一的失败点是保真度。这正是「光有闸门不够、必须同时校验输出」的实证——
如果没有保真度测试，这两份代码都能拿满分。

只要有一个该挂的没挂，任务就是无效的——**先修任务，再看 agent 表现**。

位置：`dev/fixtures/`（只用于开发，不进任何容器）。
`run_grader.py --pipeline <夹具>` 即可复现任意一行。

> 跑完记得删 `_arm_*` 目录：每个含一个 503 MB 的 `frames.u8`。


