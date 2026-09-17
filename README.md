# harbor-tasks — 显存预算型推理优化任务

把「8GB 笔记本显卡上做视频扩散推理优化」里**可移植、可判定**的部分，
做成 Harbor / Terminal-Bench 格式的任务族。

核心可复用件是一个**显存预算闸门**：verifier 在硬性 allocator 上限下重跑 agent 的代码，
用进程外的 NVML 独立测峰值，并校验输出保真度。它可靠地回答
「峰值显存有没有超过 8 GiB」——以及明确地**不**回答「真实 8GB 消费卡上一定够快」
（边界见 `docs/VRAM_GATE_DESIGN_CN.md` §4）。

---

## 目录

```
harbor-tasks/
├── README.md                        # 本文件
├── docs/
│   ├── HARBOR_TASK_FORMAT_CN.md     # 官方格式规范逐条摘要（对齐 22 个校验脚本）
│   ├── VRAM_GATE_DESIGN_CN.md       # 闸门设计 + 保真度边界 + 踩过的三个坑
│   └── TASK_FAMILY.md               # 任务族清单、死路登记、推进顺序
├── tasks/
│   └── wan-vram-budget/             # 第一个任务（已跑通四臂验证 + 22 项静态检查全绿）
│       ├── task.toml
│       ├── instruction.md
│       ├── README.md
│       ├── environment/{Dockerfile,app/}
│       ├── solution/{solve.sh,oracle_pipeline.py}
│       └── tests/{Dockerfile,test.sh,test_outputs.py,budget_harness.py,vram_sampler.py,reference/}
└── dev/                             # 仅开发用，不进任何容器
    ├── smoke.py                     # parity / budget 两档自检
    ├── run_grader.py                # 无 pytest 时的 grader 驱动器（8 个测试全调）
    ├── run_checks.sh                # 批量跑上游 check-*.sh
    ├── probe_decode_range.py        # 量解码阶段的数值分布
    └── fixtures/                    # 作弊臂夹具
```

---

## 当前状态

| 项 | 状态 |
|---|---|
| 官方格式规范 | ✅ 已抓取并逐条对齐 **22 个** `check-*.sh`（从完整 clone 中取得，非摘要） |
| GPU 类型白名单 | ✅ 已确认（**不只 H100**，含 T4/L4/A10/A100 等） |
| `wan-vram-budget` 任务骨架 | ✅ 14 个文件 |
| 静态检查 | ✅ **22 项全部 PASS，0 FAIL**（在 `terminal-bench/` 内按相对路径跑） |
| 真实 pytest grader | ✅ 在 RTX 5060 Laptop 8GB 上跑通 |
| 四臂验证（oracle / starter / 两个作弊臂） | ✅ 见下 |
| canary / slug / 作者字段 | ✅ 已按检查脚本要求修正 |
| `environment/Dockerfile` 与 `tests/Dockerfile` | ⚠️ **已写未构建**——tag / digest / wheel 已通过 registry API 验证存在，但镜像本身未拉取构建 |

### 四臂验证结果（本机 RTX 5060 Laptop 8GB，真实 pytest 9.1.1，8 个测试）

| 臂 | 做了什么 | 通过 | reward | 峰值显存 |
|---|---|---|---|---|
| `oracle` | 正确且省显存的参考解 | **8/8** | **1.0** | 1.66 GiB |
| `starter` | 未改动的起始代码 | 4/8 | 0.0 | 9.38 GiB（OOM） |
| `approx_wrong` | 显存合规但把 gain 从 0.5 扰动到 0.2 | 7/8 | 0.0 | 1.66 GiB |
| `skip_dit` | 直接 `return x` 删掉整个注意力栈 | 7/8 | 0.0 | 1.30 GiB |

两个作弊臂的失败点都精确落在保真度上（mean|diff| = 10.99 / 13.76，阈值 2.0），
说明**显存闸门不是唯一判据**，不能靠「少算点东西」通关。

另外测过一个中间态：**只修注意力、保留整卷 decode → 仍然 OOM 在 10.35 GiB**。
这证实了题面里「只修一处不够」的说法，而不是断言它。

---

## 怎么跑

### 只做自检（不需要 docker、不需要 harbor）

```bash
PY="/c/Python314/python.exe"        # 任何带 torch+CUDA 的 python

# 合法性：起始实现与参考解在同一问题规模下必须数值一致
"$PY" dev/smoke.py parity

# 显存：三臂（oracle 必须过 / 半修必须挂 / 起始必须挂）
"$PY" dev/smoke.py budget
```

### 跑真 grader（四臂之一）

```bash
"$PY" dev/run_grader.py \
  --pipeline "D:/codebase/multimodal/harbor-tasks/tasks/wan-vram-budget/tests/reference/oracle_pipeline.py" \
  --out-dir  "D:/codebase/multimodal/harbor-tasks/_arm_oracle"
```

`run_grader.py` 会依次调用 `test_outputs.py` 里全部 **8 个**测试并打印 partial score。
它和 `test_outputs.py` 的测试清单**必须保持同步**——历史上漏调过一个自检测试，
导致本地 7/7 全绿而 CI 会跑 8 个（见 `HARBOR_TASK_FORMAT_CN.md` §10）。

用 `TB_AGENT_PIPELINE` 与 `TB_AGENT_OUT_DIR` 两个**仅开发用**的环境变量切换被测对象
（容器里不设这两个变量，走 `/app` 固定路径）。

> 每次跑完记得删掉 `_arm_*` 目录：每个含一个 503 MB 的 `frames.u8`，几个臂就是几个 GB。

### 静态检查（**必须把任务复制进 `terminal-bench/tasks/`**）

```bash
git clone https://gh-proxy.com/https://github.com/harbor-framework/terminal-bench.git
SRC=harbor-tasks/tasks/wan-vram-budget
DST=terminal-bench/tasks/wan-vram-budget
rm -rf "$DST" && cp -r "$SRC" "$DST"
cd terminal-bench
for c in scripts/checks/check-*.sh; do
  bash "$c" tasks/wan-vram-budget > /tmp/o 2>&1 \
    && echo "PASS $(basename "$c")" \
    || { echo "FAIL $(basename "$c")"; cat /tmp/o; }
done
```

**不要**把仓库外的绝对路径直接传给检查脚本。脚本内部大量用
`find tasks ... -path "*/environment/Dockerfile"` 这类相对路径 + glob，
传绝对路径会得到一片**假绿**——脚本报「No files to check」然后 `exit 0`。
我第一次跑就踩了：24 个里 16 个是假的 SKIP。

### 完整流程（需要 `harbor` + docker 或 Modal）

```bash
uv tool install "harbor[modal,daytona]"
harbor run -p tasks/wan-vram-budget --agent oracle   # 必须 1.0
harbor run -p tasks/wan-vram-budget --agent nop      # 必须 0.0
```

本机有真实 8GB 卡，用 `-e docker` 本地跑**约束是真的**，可以完全绕开 Modal。
`harbor tasks start-env -p tasks/wan-vram-budget -e docker -a -i` 能进容器调试。

---

## 未完成的事（明确列出，不假装已完成）

1. **两个 Dockerfile 都没构建过。** 已通过 Docker registry API 与 PyTorch wheel
   索引**验证过三项元数据确实存在**（无需拉取几 GB 镜像）：
   - `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` ✅ 存在，digest
     `sha256:6e93a6558a7fc4fbcbf967e5fc2ea9e104df8c2e294dfcfd97fc0be48ab7d75d`（已写进 FROM）
   - `cu130` 索引有 `torch-2.14.0%2Bcu130-cp312-...manylinux_2_28_x86_64.whl` ✅
   - 两个 Dockerfile 均已通过 `check-dockerfile-sanity.sh`
   
   但**镜像本身从未构建运行**。本机出口带宽约 230 KB/s，拉几个 GB 的 CUDA 镜像不现实。
2. **没跑过 `harbor check` / `harbor run`**，因为 `uv`/`harbor` 尚未安装。
   静态检查已全绿，但端到端 trial 未跑。
3. **保留的参考快照 `../_harbor_ref/`** 只是抓取产物，可随时删除；
   现在有了完整的 `terminal-bench` clone，它以相对路径提供更权威的参考。

---

## 下一步建议

按 `docs/TASK_FAMILY.md` §5：先做 **T2 `attention-silent-fallback`**——
它是「静默降级」分类学的直接载体，闸门零件本任务已全部写好，可直接复用。
（注意族里所有 slug 都已按 `check-task-slug.sh` 的 3 词上限校准。）
