# 参与开发指南（CONTRIBUTING）

感谢你愿意改进这个项目。本文说明如何搭建环境、项目的代码约定、以及提交改动的流程。

## 目录

- [行为准则](#行为准则)
- [快速开始](#快速开始)
- [开发约定](#开发约定)
- [提交前必须通过的检查](#提交前必须通过的检查)
- [提交信息规范](#提交信息规范)
- [Pull Request 流程](#pull-request-流程)
- [常见改动怎么做](#常见改动怎么做)
- [不要做的事](#不要做的事)

## 行为准则

- 讨论聚焦技术问题，对事不对人。
- 不要提交任何真实的个人信贷数据、身份证号、手机号或未脱敏的业务数据。
- 不要引入需要付费授权才能使用的数据集或字体。

## 快速开始

本项目运行时**只依赖 Python 标准库**，因此不需要创建虚拟环境也能跑起来。

```bash
git clone https://github.com/Nausicaa613/credit-risk-scoring-service.git
cd credit-risk-scoring-service

# 1. 生成合成数据（固定随机种子，结果可复现）
python scripts/generate_dataset.py --rows 6000 --seed 20260101 --output data/applications.jsonl

# 2. 训练模型并产出报告
python scripts/train_model.py --data data/applications.jsonl --model models/model.json

# 3. 跑测试
python -m unittest discover -s tests -t . -v

# 4. 端到端自检（在临时目录中跑完整链路，不污染工作区）
python scripts/smoke_check.py

# 5. 启动服务（Makefile / run.ps1 的 run 目标会自动设置 PYTHONPATH=src）
python -m riskscore.server
```

Windows 上没有 GNU make 时，用等价脚本：

```powershell
.\run.ps1 data
.\run.ps1 train
.\run.ps1 test
.\run.ps1 smoke
.\run.ps1 run
```

`run.ps1` 与 `Makefile` 的目标名和默认参数完全一致，改其中一个时请同步改另一个。

## 开发约定

### 语言与风格

- **代码与注释：英文。** 变量名、函数名、docstring 一律英文，便于开源协作。
- **面向用户的文档：中文为主**，`README.en.md` 提供英文版；两者内容需保持同步。
- 注释解释**为什么**，而不是复述代码在做什么。反例：`# 自增 1`。
- 每个模块、每个公开函数都要有 docstring。涉及取舍的地方，在 docstring 里写清被放弃的方案和原因。
- 行宽控制在 100 字符以内。

### 分层与依赖方向

依赖只能从外向内，不允许反向：

```text
server.py  ->  api.py  ->  scorecard.py  ->  model.py  ->  features.py
                    \->  storage.py / audit.py / metrics.py
config.py、errors.py 被所有层依赖，且自身不依赖任何业务模块
```

- 不要在 `model.py` 里引入数据库或 HTTP 概念。
- 不要在 `api.py` 里直接操作 socket；`api.py` 必须能在不绑定端口的情况下被测试。
- 新增第三方依赖需要非常强的理由。运行时的零依赖是本项目的核心卖点，CI 会检查 `src/` 下是否出现了非标准库导入。

### 测试

- 测试用标准库 `unittest`，位置在 `tests/`，文件名 `test_<模块>.py`。
- 新增功能必须同时新增测试；修 bug 必须先写一个能复现该 bug 的测试。
- 测试之间必须互相独立：不得依赖执行顺序，不得写入真实的工作目录（使用 `tempfile`）。
- 涉及随机性的地方一律用**固定种子**，禁止依赖 `random` 的默认种子。
- 断言要针对行为而非实现细节。例如断言"贡献度之和等于 log-odds"，而不是断言某个权重等于某个具体小数。

## 提交前必须通过的检查

```bash
python -m unittest discover -s tests -t . -v   # 全部通过
python scripts/smoke_check.py                  # 端到端通过
python -m compileall -q src scripts tests      # 无语法错误
```

或者一条命令：`make check`（等价于 `.\run.ps1 check`）。

CI 会在 Python 3.9 / 3.10 / 3.11 / 3.12 以及 Windows、macOS 上重复这些检查，并额外验证容器可以构建、启动并通过 `/healthz`。本地只跑一个版本的 Python 时，请避免使用该版本特有的语法。

## 提交信息规范

采用 Conventional Commits 风格，一行标题，必要时补充正文说明动机：

```text
feat(scorecard): derive band edges from the published cut-offs
fix(storage): stop seeding the audit trail from init_db
docs(readme): document the calibration split
test(api): cover the 413 path for oversized bodies
refactor(model): extract the prior-offset fit into its own function
perf(metrics): cap the score observation buffer
```

标题用祈使句、现在时、不超过 72 字符。正文里说明**为什么要改**以及**怎么验证的**。

## Pull Request 流程

1. 从 `main` 切出分支：`git checkout -b fix/band-edge-drift`。
2. 保持改动聚焦，一个 PR 只解决一件事。
3. 确保 `make check` 全绿。
4. 在 PR 描述里写清：问题是什么、怎么改的、怎么验证的、有哪些已知取舍。
5. 如果改动了模型行为（特征、评分公式、分档、阈值），必须在 PR 里附上重新训练后的指标对比，并同步更新 `MODEL_CARD.md`。
6. 如果改动了 API 契约，必须同步更新 `README.md`、`README.en.md` 与 `docs/DESIGN.md`。

## 常见改动怎么做

### 新增一个特征

1. 在 `src/riskscore/features.py` 的 `NUMERIC_FIELDS` 中加入字段及其取值范围。
2. 如果这个特征是金额类的长尾分布，把它加入 `_LOG1P_FIELDS`；如果是比例类，考虑加入 `_CLIPPED_FIELDS`。
3. 在 `src/riskscore/scorecard.py` 的 `FEATURE_SEMANTICS` 中补上描述和预期的风险方向，否则原因码会退化成原始字段名。
4. 在 `scripts/generate_dataset.py` 中生成该字段，并在 `_latent_logit` 中体现它与风险的关系。
5. 更新 `tests/test_features.py` 中 `FEATURE_NAMES` 的长度断言。
6. 重新训练并更新 `MODEL_CARD.md` 的特征表。

注意：特征顺序是模型契约的一部分。`model.json` 会记录训练时的特征列表，顺序不一致时服务会直接拒绝加载，这是刻意设计。

### 换一个模型

`ScorecardModel` 对外只暴露 `predict_proba`、`predict`、`contributions` 和 `assert_compatible`。只要新模型实现同样的接口，`Scorer` 和整个 API 层都不需要改。替换后请保留 `contributions`，否则可解释性会丢失，原因码将无法生成。

### 新增一个存储驱动

`ApplicationStore` 和 `AuditLog` 是存储的边界。新增驱动时保持方法签名不变，并在 `tests/` 中加入对应的集成测试。注意保持审计表的**只追加**语义：不要为它新增 update/delete 方法。

### 调整评分刻度或分档

分档边界是从概率切点**计算**出来的（见 `scorecard.py` 的 `_BAND_CUTOFFS` 与 `_build_risk_bands`），不要硬编码分数。修改切点后，`tests/test_scorecard.py` 中的独立性副本会失败，请一并更新并把新的 PD–分数对应关系写进 docstring。

## 不要做的事

- 不要为了让测试通过而放宽断言，尤其是与风险排序方向、分数单调性、贡献度可加性相关的断言。
- 不要把 `data/`、`models/`、`reports/` 提交进仓库：它们可由固定种子完全复现，且会让仓库体积膨胀。
- 不要在请求路径中让审计写入失败导致打分失败（当前策略是记录日志并继续），也不要悄悄改成静默丢弃。
- 不要移除 `Scorer` 的分数裁剪与分档一致性约束。
- 不要声称本项目可用于真实信贷决策。它是作品集与教学项目，数据是合成的。
