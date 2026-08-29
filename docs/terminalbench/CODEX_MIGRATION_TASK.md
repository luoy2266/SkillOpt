# Codex 任务：将 SkillOpt 迁移到 Terminal-Bench v2.1（Harbor + Terminus-2 + Docker + DeepSeek-V4-Flash-0731）

你现在需要完成一个研究代码迁移任务：

> 基于 Microsoft 官方 SkillOpt 仓库，在尽量不修改 SkillOpt 核心算法的前提下，新增对 Terminal-Bench v2.1 的 benchmark/environment 支持，使 SkillOpt 能通过 Harbor + Terminus-2 + Docker 使用 DeepSeek-V4-Flash-0731 执行 Terminal-Bench rollout，并将 verifier reward 与 trajectory 回流到 SkillOpt 原有的 reflection / update / validation gate 流程中。

最终目标不是在本机完成大规模训练，而是开发一个**可以交付给项目负责人，在服务器上完成全量训练和评测的迁移仓库**。

本地开发阶段只要求：

1. 单元测试和 mock 流程通过；
2. Harbor + Docker 的真实单任务 rollout 能运行；
3. 至少跑通一次完整 SkillOpt 闭环：

```text
train rollout
→ verifier reward
→ trajectory
→ reflection
→ candidate skill
→ validation rollout
→ gate
```

不要求 smoke test 中性能提升。

---

# 一、必须遵守的总体原则

## 1. 基于官方 SkillOpt 原仓库开发

使用 Microsoft 官方 SkillOpt：

```text
https://github.com/microsoft/SkillOpt
```

开发开始时：

1. 拉取当前最新 `main`；
2. 立即记录完整 commit SHA；
3. 从该 commit 创建迁移分支；
4. 此后迁移过程中不要继续跟随 upstream main 漂移。

例如：

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt

git checkout main
git pull

git rev-parse HEAD

git checkout -b terminalbench-v2.1
```

创建：

```text
UPSTREAM.md
```

至少记录：

```text
Upstream repository
Base branch
Base commit SHA
Migration branch
Migration date
```

不要使用浮动的“latest”作为最终实验 provenance。

---

## 2. 不修改 SkillOpt 核心训练算法

除非确有必要，不修改：

```text
trainer
reflection algorithm
aggregation
selection
skill update
validation gate
optimizer logic
```

本次工作的主要扩展点应该是：

```text
skillopt/envs/terminalbench/
```

按照 SkillOpt 官方：

```text
docs/guide/new-benchmark.md
```

定义的 benchmark extension contract 实现。

优先复用：

```text
SplitDataLoader
EnvAdapter
shared reflection
原 train CLI
原 eval_only CLI
原 config system
```

不要重新实现一套 SkillOpt trainer。

---

## 3. 不修改 Terminal-Bench

不要 fork 或修改：

```text
Terminal-Bench tasks
Dockerfiles
tests
verifier
solutions
```

Terminal-Bench verifier 是唯一权威 scorer。

不要自己通过 final answer 文本判断任务是否成功。

---

## 4. Harbor 是外部依赖

Harbor 不 fork、不 vendor、不复制进 SkillOpt repository。

暂时固定：

```text
Harbor = 0.20.0
```

迁移 repository 只负责：

```text
检查 Harbor
调用 Harbor
生成 Harbor config
读取 Harbor result
读取 Harbor trajectory
```

不要直接修改 Harbor 源码。

如果未来确实需要 Harbor patch，应采用：

```text
patch file
+ version check
+ hash/provenance
```

的形式，而不是维护 Harbor fork。

---

## 5. 执行环境固定使用 Docker

最终 target rollout 必须是：

```text
SkillOpt
    ↓
TerminalBenchAdapter
    ↓
Harbor 0.20.0
    ↓
Terminus-2 v2.0.0
    ↓
Docker
    ↓
DeepSeek-V4-Flash-0731
    ↓
Terminal-Bench v2.1
```

---

# 二、模型配置

本实验有两个逻辑角色，但模型均使用：

```text
DeepSeek-V4-Flash-0731
```

必须始终区分两条调用链。

## Optimizer model

SkillOpt reflection / aggregation / skill update 使用：

```text
DeepSeek-V4-Flash-0731
```

可以通过 SkillOpt 已有的 OpenAI-compatible / provider-compatible backend 接入。

调用逻辑：

```text
SkillOpt optimizer
→ API
→ DeepSeek-V4-Flash-0731
```

---

## Target model

Terminal-Bench task 执行不能由 SkillOpt 直接调用模型。

必须：

```text
SkillOpt
→ Harbor
→ Terminus-2
→ DeepSeek-V4-Flash-0731
```

也就是说：

> TerminalBenchAdapter 不应该调用 `chat_target()` 来完成 Terminal-Bench task。

Terminal-Bench 属于 environment-specific / exec-style rollout。

---

# 三、实验公平性硬约束

最终需要公平比较：

```text
Baseline:
skills = []

SkillOpt:
skills = [generated SkillOpt skill]
```

baseline 和 SkillOpt 必须使用同一个 Harbor runner 和同一个 agent template。

必须保持一致：

```text
same model
same Terminus-2
same Harbor
same Docker backend
same Terminal-Bench tasks
same Docker images
same CPU
same memory
same disk
same GPU
same timeout
same max turns
same reasoning config
same retry policy
same proxy
same cache
same concurrency policy
same task instruction/template
```

唯一主要实验变量：

```text
skills
```

Skill 必须通过 Harbor 原生：

```yaml
agents:
  - skills:
      - ...
```

机制注入。

禁止：

```text
手工把 skill 拼到 system prompt
修改 Terminus-2 system prompt
为 SkillOpt 使用另一套 agent
```

---

# 四、Terminal-Bench 数据划分

当前暂时采用：

```text
train : val : test = 1 : 1 : 8
```

89 个任务可暂时划分为：

```text
train = 9
val   = 9
test  = 71
```

具体 task ID 不需要严格对齐主方法。

项目负责人拿到 repository 后会自行调整。

因此必须把 split 做成**可替换 manifest**，不能硬编码在 adapter 中。

推荐：

```text
data/
└── terminalbench_split/
    ├── train/
    │   └── tasks.json
    ├── val/
    │   └── tasks.json
    └── test/
        └── tasks.json
```

同时提供：

```text
scripts/materialize_terminalbench_split.py
```

例如：

```bash
python scripts/materialize_terminalbench_split.py \
  --ratio 1:1:8 \
  --seed 42
```

输出：

```text
manifests/terminalbench_split.json
manifests/terminalbench_split.sha256
```

数据文件只保存 Terminal-Bench task identifiers 和必要 metadata。

不要复制：

```text
task solution
test implementation
Dockerfile
完整 benchmark 数据副本
```

---

# 五、先阅读并理解 SkillOpt 官方协议

开始实现前，必须首先认真阅读：

```text
docs/guide/new-benchmark.md
```

然后至少检查：

```text
skillopt/envs/base.py
skillopt/datasets/base.py
scripts/train.py
scripts/eval_only.py
SkillOpt training loop
SkillOpt reflection pipeline
SkillOpt validation gate
现有 exec-style benchmark/environment adapter
```

重点确认：

```text
EnvAdapter abstract methods
SplitDataLoader contract
rollout return schema
conversation.json schema
reflection 如何加载 trajectory
train/eval registry
config inheritance
skill_init 行为
RolloutResult extras
```

不要在没有理解 upstream contract 的情况下直接重写框架。

---

# 六、先写协议文档

编码前创建：

```text
docs/terminalbench/PROTOCOL_MAPPING.md
```

明确记录以下映射。

## Dataset

```text
SkillOpt:
SplitDataLoader / train-val-test

Terminal-Bench:
task ID manifest
```

---

## Skill

```text
SkillOpt:
skill_content: str

Harbor:
skill directory
└── SKILL.md
```

---

## Execution

```text
SkillOpt:
EnvAdapter.rollout()

Terminal-Bench:
Harbor
→ Terminus-2
→ Docker
→ DeepSeek-V4-Flash-0731
```

---

## Reward

SkillOpt rollout 需要至少返回：

```text
id
hard
soft
```

Terminal-Bench 使用 Harbor verifier reward。

第一版：

```python
hard = reward
soft = float(reward)
```

不要实现自定义 scorer。

---

## Trajectory

SkillOpt reflection 需要：

```text
<rollout_dir>/
└── predictions/
    └── <task-id>/
        └── conversation.json
```

Harbor/Terminus-2 产生 ATIF trajectory。

需要：

```text
ATIF
→ SkillOpt-compatible conversation.json
```

---

# 七、推荐新增的目录

推荐最终新增：

```text
skillopt/
└── envs/
    └── terminalbench/
        ├── __init__.py
        ├── adapter.py
        ├── dataloader.py
        ├── rollout.py
        ├── harbor_runner.py
        ├── result_parser.py
        ├── trajectory.py
        ├── skill_pack.py
        └── skills/
            └── initial.md
```

职责必须严格分离。

不要把所有逻辑塞进 `adapter.py`。

---

# 八、实现 dataloader

文件：

```text
skillopt/envs/terminalbench/dataloader.py
```

职责：

```text
读取 train/val/test task ID
构造 SkillOpt train batch
构造 SkillOpt eval batch
提供 task metadata
```

不要：

```text
调用 Harbor
启动 Docker
解析 reward
读取 trajectory
```

优先遵守 SkillOpt 官方 `SplitDataLoader` 接口。

---

# 九、实现 skill_pack.py

文件：

```text
skillopt/envs/terminalbench/skill_pack.py
```

输入：

```python
skill_content: str
```

输出 Harbor-compatible skill directory，例如：

```text
<output_root>/
└── harbor_skills/
    └── <sha256>/
        └── terminalbench-skill/
            └── SKILL.md
```

要求：

1. SKILL.md 的主要 body 应与 SkillOpt candidate skill 内容一致；
2. 不二次 summarize；
3. 不插额外 TBench prompt；
4. 不加入 task-specific solution；
5. 记录 SHA-256；
6. 相同 skill_content 应尽量产生稳定 digest。

创建：

```text
skillopt/envs/terminalbench/skills/initial.md
```

初始 skill 默认 blank。

---

# 十、实现 HarborRunner

文件：

```text
skillopt/envs/terminalbench/harbor_runner.py
```

这是迁移的核心执行模块。

职责：

```text
输入：
task IDs
skill paths
runtime config
result name

↓
生成 Harbor job config

↓
执行 Harbor

↓
等待任务结束

↓
收集 trial result / verifier / ATIF path
```

HarborRunner 不负责：

```text
SkillOpt reflection
Skill update
validation gate
```

---

## HarborRunner 必须同时支持 baseline 和 Skill

例如统一接口：

```python
runner.run(
    task_ids=[...],
    skills=[],
)
```

以及：

```python
runner.run(
    task_ids=[...],
    skills=[skill_dir],
)
```

不要维护：

```text
baseline_runner.py
skill_runner.py
```

两套不同代码。

---

# 十一、Harbor config generator

建议 HarborRunner 内部或单独模块实现统一 config builder。

baseline：

```yaml
agents:
  - ...
    skills: []
```

SkillOpt：

```yaml
agents:
  - ...
    skills:
      - /path/to/generated/skill
```

两份 resolved config 除以下内容外应一致：

```text
skills
result_name
output path
skill provenance/hash
```

必须保存 resolved config。

---

# 十二、实现 result_parser.py

文件：

```text
skillopt/envs/terminalbench/result_parser.py
```

从 Harbor trial result 中读取：

```text
verifier reward
trial status
result path
```

转换为 SkillOpt rollout result。

例如：

```python
{
    "id": task_id,
    "hard": reward,
    "soft": float(reward),
    "task_type": "terminalbench",
    "trial_status": ...,
    "harbor_result_path": ...,
    "infrastructure_valid": ...,
}
```

不要把：

```text
reward=0
```

等价成：

```text
infrastructure failure
```

真实模型失败与基础设施失败必须可区分。

不要根据 reward 自动重跑。

---

# 十三、实现 trajectory.py

文件：

```text
skillopt/envs/terminalbench/trajectory.py
```

目标：

```text
Harbor / Terminus-2 ATIF trajectory
        ↓
SkillOpt conversation.json
```

输出必须符合 SkillOpt reflection 读取要求：

```text
<rollout_dir>/
└── predictions/
    └── <task-id>/
        └── conversation.json
```

第一版尽可能保留：

```text
task instruction
assistant/agent messages
reasoning text（若 SkillOpt schema 合法且 upstream 允许）
tool calls
terminal commands
terminal outputs
environment observations
final response
```

如果 SkillOpt shared reflection 只接受普通 role/content message，则把 terminal interaction 安全序列化为文本，例如：

```text
[TOOL]
command: ...

[OBSERVATION]
...
```

优先适配 trajectory。

不要首先修改 SkillOpt reflection algorithm。

---

# 十四、实现 rollout.py

文件：

```text
skillopt/envs/terminalbench/rollout.py
```

执行链：

```text
items
+ skill_content

↓
skill_pack

↓
HarborRunner

↓
result_parser

↓
trajectory converter

↓
SkillOpt rollout results
```

伪代码：

```python
def run_batch(
    *,
    items,
    skill_content,
    out_root,
    runtime_config,
    ...
):
    skill_path = package_skill(
        skill_content=skill_content,
        out_root=out_root,
    )

    harbor_results = runner.run(
        task_ids=[item["id"] for item in items],
        skills=[skill_path],
    )

    results = []

    for item in items:
        trial = harbor_results[item["id"]]

        result = parse_trial_result(trial)

        convert_atif_to_conversation(
            trial=trial,
            output_path=...,
        )

        results.append(result)

    return results
```

注意：

> TerminalBench rollout 不允许直接绕过 Terminus-2 调 DeepSeek。

---

# 十五、实现 TerminalBenchAdapter

文件：

```text
skillopt/envs/terminalbench/adapter.py
```

Adapter 本身保持薄。

按照 SkillOpt 当前 `EnvAdapter` API 实现所要求的方法，例如：

```text
build_train_env
build_eval_env
rollout
get_task_types
```

具体函数签名必须以 pinned SkillOpt commit 的真实 API 为准，不要凭假设编写。

结构应类似：

```text
setup
→ dataloader

build_train_env
→ dataloader

build_eval_env
→ dataloader

rollout
→ terminalbench.rollout.run_batch

get_task_types
→ terminalbench
```

---

# 十六、保持 SkillOpt 原 CLI/config

不要新建独立训练框架。

继续支持 SkillOpt 原本：

```bash
python scripts/train.py --config ...
```

以及：

```bash
python scripts/eval_only.py --config ...
```

根据 pinned upstream 的 registry 机制，在 `train.py` 和 `eval_only.py` 中注册：

```text
terminalbench
```

应采用 upstream 当前推荐的 lazy registration 方式。

不要引入不必要的全局 registry 重构。

---

# 十七、配置结构

新增：

```text
configs/
└── terminalbench/
    ├── default.yaml
    ├── smoke.yaml
    └── full.yaml
```

## default.yaml

保存实验语义，例如：

```text
benchmark = terminalbench
optimizer model = DeepSeek-V4-Flash-0731
target model identity = DeepSeek-V4-Flash-0731
Harbor = 0.20.0
agent = Terminus-2 v2.0.0
environment = Docker
skill_init = blank
split_dir = ...
```

---

## smoke.yaml

只缩小：

```text
task 数
epoch
batch size
workers/concurrency
```

不要改变核心执行协议。

目标：

```text
1 train task
1 val task
1 epoch
1 optimization step
```

---

## full.yaml

给项目负责人服务器运行。

不要在其中写死负责人未来一定要使用的：

```text
API key
机器绝对路径
GPU 数
并发数
cache 绝对路径
```

这些应支持环境变量或 CLI override。

---

# 十八、先实现 mock mode

在真实调用 Harbor 前，先完成 lightweight mock。

例如：

```yaml
env:
  mock: true
```

mock rollout 返回：

```text
固定 reward
固定 trajectory
```

测试：

```text
SkillOpt trainer
→ TerminalBenchAdapter
→ rollout
→ reflection
→ candidate skill
→ validation
→ gate
```

目标是发现：

```text
config bug
schema bug
path bug
registry bug
reflection artifact bug
```

而不是烧模型 token。

---

# 十九、真实 smoke test 1：single rollout

Mock 通过后，第一次真实调用：

```text
Harbor 0.20.0
+ Terminus-2
+ Docker
+ DeepSeek-V4-Flash-0731
```

只跑一个 task。

验证：

```text
SkillOpt skill_content
→ SKILL.md
→ Harbor agents[].skills
→ trial sandbox
→ Terminus-2
→ Docker
→ task
→ verifier reward
→ ATIF
→ conversation.json
```

不要求任务 pass。

---

# 二十、真实 smoke test 2：baseline / skill parity

固定一个 task。

运行：

```text
A:
skills=[]

B:
skills=[fixed_test_skill]
```

保存 resolved configs。

实现自动 parity check。

例如：

```text
tests/terminalbench/test_config_parity.py
```

允许差异：

```text
skills
result_name
output directory
skill hash/provenance
```

若发现：

```text
model
agent
Docker resources
timeout
reasoning
retry
prompt/template
```

不同，则测试失败。

---

# 二十一、真实 smoke test 3：完整 SkillOpt 闭环

本地最终验收要求：

```text
1 train task
1 val task
1 epoch
1 optimization step
```

必须真实完成：

```text
blank initial skill
        ↓
train rollout
        ↓
verifier reward
        ↓
ATIF
        ↓
conversation.json
        ↓
SkillOpt reflection
        ↓
aggregate/select/update
        ↓
candidate skill
        ↓
validation rollout
        ↓
validation reward
        ↓
gate
        ↓
accept / reject
        ↓
best skill / artifact
```

即使：

```text
new reward < old reward
```

也没有关系。

工程验收标准是：

> pipeline 正确闭环。

不是：

> smoke test 一定提升。

---

# 二十二、必须测试 eval_only

训练闭环通过后，测试 SkillOpt 原：

```text
scripts/eval_only.py
```

至少支持：

```text
baseline
```

和：

```text
generated/best skill
```

例如最终负责人应能做到：

```text
训练 best_skill
↓
跑 baseline
↓
跑 best_skill
↓
比较 held-out test
```

而不需要修改迁移代码。

---

# 二十三、Docker / Harbor preflight

新增：

```text
scripts/preflight_terminalbench.py
```

该脚本只负责检查，不要自动修改宿主系统。

至少检查：

```text
Docker daemon available
当前用户 Docker 权限
Docker disk space
Harbor executable
Harbor version == 0.20.0
Terminus-2 availability/config
Terminal-Bench v2.1 task path
DeepSeek API endpoint
DeepSeek credentials
proxy variables
NO_PROXY
cache directories
PYTHONPATH
```

如果条件不满足：

```text
fail closed
```

不要偷偷降级。

---

# 二十四、Docker 负责人踩坑经验

以下经验需要纳入交付 repository，但不要全部写入 EnvAdapter。

## 地址池

正式大规模运行可能需要：

```text
/etc/docker/daemon.json
```

扩展 default address pools。

迁移代码只检查/文档提示。

不要自动修改 `/etc/docker/daemon.json`。

---

## Cleanup

提供：

```text
scripts/cleanup_terminalbench.py
```

只定向清理 Terminal-Bench/Harbor 对应的：

```text
__env-main containers
__env_default networks
```

不要执行：

```bash
docker system prune -a
```

这种破坏性全局清理。

---

## Docker group / systemd

负责人服务器长期运行可能通过：

```text
systemd --user
```

启动。

文档中提醒必要时使用：

```bash
sg docker -c '...'
```

确保进程拥有 Docker group 权限。

这属于：

```text
docs/terminalbench/SERVER_RUNBOOK.md
```

而不是 SkillOpt core。

---

# 二十五、cache 支持

Harbor/Docker runtime config 应支持负责人现有公共缓存。

包括：

```text
Hugging Face
pip
uv
DistilBERT
Qwen tokenizer
OpenThoughts
CIFAR-10
QEMU
POV-Ray
```

公共静态缓存应支持：

```text
read-only mount
MANIFEST.tsv
SHA-256
```

不要把大型 cache 文件提交进 Git repository。

只保存：

```text
配置
路径 contract
manifest validation logic
文档
```

---

# 二十六、代理

支持向 task container 注入：

```text
HTTP_PROXY
HTTPS_PROXY
http_proxy
https_proxy
```

同时支持：

```text
NO_PROXY
no_proxy
```

localhost / 127.0.0.1 / ::1 / Docker internal addresses 不应被错误送到公网代理。

proxy 行为必须 baseline / SkillOpt 完全一致。

---

# 二十七、资源配额和 cache prompt 提示

负责人以前可能在 task instruction 中额外告诉 agent：

```text
CPU
RAM
disk
GPU
cache path
```

本迁移 repository 不要主动改变这一行为。

原则：

```text
如果负责人现有 Docker baseline 使用这些 prompt augmentation
→ baseline 和 SkillOpt 都继承

如果现有 baseline 没用
→ migration 不自行新增
```

不要为了 SkillOpt 额外改变 agent prompt。

---

# 二十八、Harbor local task config

保持 dataset/task source config 尽量最小。

不要为了兼容历史 Harbor bug 添加不必要 metadata。

如：

```text
本地显式 tasks
```

优先只声明真实需要的：

```text
path
```

具体 schema 以 Harbor 0.20.0 实际实现为准。

---

# 二十九、result_name 和 provenance

不同条件必须使用不同结果名，例如：

```text
tbench_baseline
tbench_skillopt_train
tbench_skillopt_eval
tbench_smoke
```

禁止不同实验结果错误合并。

每次正式运行保存：

```text
resolved SkillOpt config
resolved Harbor config
task manifest
task manifest checksum
SkillOpt upstream commit
Harbor version
Terminus-2 version
Docker environment identity
model request name
underlying model = DeepSeek-V4-Flash-0731
skill SHA-256
result name
output paths
```

---

# 三十、请求级可靠性

如果已有 provider/API wrapper，则实现或保留有界 transient retry。

可以重试：

```text
network error
429
部分 5xx
已确认 transient backend overflow
```

不要：

```text
无限 retry
根据 reward retry
把程序 bug 当网络错误 retry
```

如果实现 API key rotation，需要记录：

```text
key index
attempt
status code
latency
wait
retry reason
```

推荐：

```text
request_events.jsonl
```

但不要为了这个功能大幅改 SkillOpt 核心。

---

# 三十一、推荐最终 repository 结构

```text
SkillOpt/
│
├── skillopt/
│   └── envs/
│       └── terminalbench/
│           ├── __init__.py
│           ├── adapter.py
│           ├── dataloader.py
│           ├── rollout.py
│           ├── harbor_runner.py
│           ├── result_parser.py
│           ├── trajectory.py
│           ├── skill_pack.py
│           └── skills/
│               └── initial.md
│
├── configs/
│   └── terminalbench/
│       ├── default.yaml
│       ├── smoke.yaml
│       └── full.yaml
│
├── data/
│   └── terminalbench_split/
│       ├── train/
│       ├── val/
│       └── test/
│
├── manifests/
│   ├── runtime.yaml
│   └── terminalbench_split.json
│
├── scripts/
│   ├── train.py
│   ├── eval_only.py
│   ├── materialize_terminalbench_split.py
│   ├── preflight_terminalbench.py
│   └── cleanup_terminalbench.py
│
├── tests/
│   └── terminalbench/
│       ├── test_dataloader.py
│       ├── test_skill_pack.py
│       ├── test_result_parser.py
│       ├── test_trajectory.py
│       ├── test_config_parity.py
│       └── test_mock_loop.py
│
├── docs/
│   └── terminalbench/
│       ├── PROTOCOL_MAPPING.md
│       ├── MIGRATION.md
│       └── SERVER_RUNBOOK.md
│
└── UPSTREAM.md
```

目录可根据 pinned upstream 实际结构小幅调整。

不要为了完全匹配这个树而破坏 SkillOpt 原本的代码组织。

---

# 三十二、严格开发顺序

请严格按照以下顺序实施，不要一开始直接跑全量 Harbor。

## Phase 0

```text
阅读 new-benchmark.md
阅读相关 SkillOpt 源码
pin upstream commit
创建 UPSTREAM.md
创建 PROTOCOL_MAPPING.md
```

验收：

```text
能够明确说明 EnvAdapter / rollout / trajectory / score / CLI contract
```

---

## Phase 1

```text
实现 split manifest
实现 dataloader
```

验收：

```text
train=9
val=9
test=71
可由负责人替换 manifest
```

---

## Phase 2

```text
实现 skill_pack.py
```

验收：

```text
skill_content
→ Harbor-compatible SKILL.md
→ stable SHA256
```

---

## Phase 3

```text
实现 Harbor config builder
实现 HarborRunner
```

验收：

```text
能构建 baseline config
能构建 skill config
```

---

## Phase 4

```text
实现 result_parser.py
```

验收：

```text
Harbor verifier reward
→ hard/soft
```

---

## Phase 5

```text
实现 trajectory.py
```

验收：

```text
ATIF
→ predictions/<task-id>/conversation.json
```

---

## Phase 6

```text
实现 rollout.py
```

验收：

```text
items + skill_content
→ Harbor
→ reward
→ trajectory
→ RolloutResult
```

---

## Phase 7

```text
实现 TerminalBenchAdapter
注册 train.py
注册 eval_only.py
```

验收：

```text
SkillOpt CLI 能识别 terminalbench
```

---

## Phase 8

```text
实现 default/smoke/full config
实现 mock mode
```

验收：

```text
不用 Harbor 也能跑 SkillOpt mock optimization loop
```

---

## Phase 9

真实单任务 smoke：

```text
fixed skill
→ Harbor
→ Terminus-2
→ Docker
→ DeepSeek
→ reward
→ trajectory
```

---

## Phase 10

baseline vs skill parity test：

```text
skills=[]
vs
skills=[fixed_skill]
```

验证其余 runtime config 完全一致。

---

## Phase 11

真实完整 SkillOpt 闭环：

```text
1 train
1 val
1 epoch
1 optimization step
```

必须完成：

```text
rollout
→ reflection
→ candidate
→ validation
→ gate
```

---

## Phase 12

测试：

```text
eval_only baseline
eval_only best_skill
```

---

## Phase 13

最后再实现：

```text
preflight
cleanup
cache/proxy support
server runbook
provenance
```

不要因为大规模服务器优化阻塞最初闭环。

---

# 三十三、测试要求

尽量给新增纯逻辑模块写单元测试。

至少覆盖：

```text
dataloader
split
skill packaging
skill hashing
Harbor config generation
baseline/skill parity
reward parser
ATIF conversion
mock rollout
```

真实 Harbor/Docker integration test 可单独标记：

```text
integration
```

不要让普通 unit test 自动消耗 DeepSeek API。

---

# 三十四、Definition of Done

最终迁移 repository 被认为完成，当且仅当以下条件全部满足。

## A

SkillOpt 原 CLI 可以：

```text
benchmark/environment = terminalbench
```

---

## B

1:1:8 split 已物化，并可由负责人替换。

---

## C

SkillOpt candidate skill 可以可靠转换为 Harbor 原生：

```text
SKILL.md
```

并通过：

```text
agents[].skills
```

注入。

---

## D

Terminus-2 真实运行：

```text
DeepSeek-V4-Flash-0731
```

完成 Docker Terminal-Bench task。

---

## E

Terminal-Bench verifier reward 能正确返回 SkillOpt：

```text
hard
soft
```

---

## F

Harbor ATIF trajectory 能转换为 SkillOpt reflection 需要的：

```text
conversation.json
```

---

## G

真实跑通一次：

```text
rollout
→ reflect
→ update
→ validation
→ gate
```

---

## H

`eval_only.py` 能分别评测：

```text
baseline skills=[]
```

和：

```text
SkillOpt best skill
```

---

## I

baseline 和 SkillOpt 共用相同 HarborRunner，并存在自动 config parity test。

---

## J

README / MIGRATION / SERVER_RUNBOOK 足够让负责人在服务器上配置：

```text
Harbor 0.20.0
Docker
DeepSeek API
proxy
cache
task split
并发/resources
```

而不需要修改迁移核心代码。

---

# 三十五、执行过程中的行为要求

1. 不要一次性重构整个 SkillOpt；
2. 每完成一个 Phase 就运行对应测试；
3. 优先最小侵入；
4. 对 upstream API 不确定时先阅读实际源码；
5. 不要猜测函数签名；
6. 不要把 Docker 运维代码塞入 EnvAdapter；
7. 不要修改 Harbor；
8. 不要修改 Terminus-2；
9. 不要修改 Terminal-Bench verifier；
10. 不要通过 prompt 拼接 Skill；
11. 不要直接绕过 Terminus-2 调 target model；
12. 不要为了 smoke test 运行全部 89 个任务；
13. 不要以性能是否提升作为工程 smoke test 的成功标准；
14. 所有关键配置和 provenance 必须保存；
15. 如果发现必须违反上述边界才能运行，先在 `MIGRATION.md` 中记录原因，再进行最小必要修改。

最终实现目标是：

> 在保持 SkillOpt 原有 trainer / optimizer / reflection / update / validation gate / CLI/config 体系的前提下，把 Terminal-Bench v2.1 实现为一个新的 environment adapter；target rollout 委托给冻结的 Harbor 0.20.0 + Terminus-2 + Docker + DeepSeek-V4-Flash-0731 栈，并把 verifier reward 和 trajectory 转换回 SkillOpt 原生接口。

