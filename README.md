# credit-risk-scoring-service 信用风险评分服务

一个**零第三方依赖**的信贷风险评分微服务：用纯 Python 标准库实现的可解释逻辑回归评分卡，通过 REST API 提供打分、落库与审计。

[English README](README.en.md) ｜ [设计文档](docs/DESIGN.md) ｜ [模型卡](MODEL_CARD.md) ｜ [参与开发](CONTRIBUTING.md)

![CI](https://github.com/Nausicaa613/credit-risk-scoring-service/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)
![Tests](https://img.shields.io/badge/tests-244%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 这个项目解决什么问题

信贷评分系统有一个和多数机器学习项目不同的约束：**它必须能被审查**。监管方、审计或信贷委员会需要回答"为什么这位申请人得到 612 分"，而且这个答案必须和模型当时实际使用的计算完全一致。

线性评分卡之所以仍是行业基线，正是因为这个问题有精确答案。模型在 log-odds 空间里是加性的：

```text
logit(PD) = 截距 + Σ (权重_i × 特征_i)
```

每一项 `权重_i × 特征_i` 就是该特征对这次决策的贡献，所以"可解释性"不是事后加上的近似，而是模型本身的性质。梯度提升大概率能拿到更高的 AUC（见路线图），代价是这份可审计性。

第二个约束是**可复现**。同一份数据、同一个随机种子，必须产出同一个模型。因此本项目的模型产物是纯 JSON（可 diff、可 code review），而不是 pickle 二进制。

## 功能

- **可解释**：每次打分返回全部 18 个特征的贡献值，且贡献之和 + 截距 **精确等于**模型使用的 log-odds（有测试断言）。
- **原因码**：自动挑选贡献最大的 3 个特征，标注它是抬高还是降低风险，并与业务先验方向比对。
- **概率校准**：类别加权训练会让概率系统性偏高，因此在校准集上拟合一个 log-odds 偏置（bias-only Platt scaling）来修正。**只改截距，不动权重**，所以排序、AUC、KS 完全不变。
- **评分刻度**：采用 points-to-double-the-odds 行业惯例，`分数 = 600 + (40 / ln2) × ln(好/坏几率)`，分数越高风险越低。
- **风险分档**：A–E 五档，边界由概率切点**计算得出**而非手写，因此档位标签与响应里的 PD 永远不会互相矛盾。
- **只追加审计**：`AuditLog` 只暴露 `append` / `tail`，代码里不存在改或删的路径。决策行与审计行在**同一个事务**里提交。
- **可观测**：`/healthz` 做存活与就绪判定，`/metrics` 输出计数器、延迟直方图与分数分布。
- **零依赖**：运行时只用标准库。CI 会解析 `src/` 的 AST，发现任何第三方导入就失败。
- **自带数据生成器**：固定种子，可复现；用二分法求解截距，使真实 PD 均值命中指定的基准违约率。

## 架构

```text
  客户端
    │  POST /v1/score  {"age": 40, ...}
    ▼
┌──────────────────────────────────────────────────────────────┐
│ server.py   ThreadingHTTPServer + 手写路由                    │
│             · 解析请求行/头部/body，强制最大体积限制            │
│             · 生成 request_id，写响应头，记录访问日志           │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ api.py      与传输层无关的服务层                                │
│             · 路由分发、查询参数校验、错误封装                  │
└───────┬──────────────────────────────────┬───────────────────┘
        ▼                                  ▼
┌────────────────────────┐      ┌──────────────────────────────┐
│ features.py            │      │ storage.py / audit.py        │
│ · 字段契约与别名        │      │ · applications（决策）        │
│ · log1p + one-hot 变换  │      │ · audit_events（只追加）      │
└───────────┬────────────┘      │ · 同一事务内提交两行           │
            ▼                   └──────────────────────────────┘
┌────────────────────────┐
│ model.py               │      风险分数 / 档位 / 原因码
│ · 逻辑回归、梯度下降    │               │
│ · 标准化、指标、JSON 持久化 │            ▼
└───────────┬────────────┘      ┌──────────────────────────────┐
            ▼                   │ scorecard.py                 │
   models/model.json            │ · PD → 分数（可逆）           │
   （纯 JSON，可 diff）          │ · 分数 → 档位                 │
                                │ · 贡献值 → 原因码             │
                                └──────────────────────────────┘
```

目录结构：

```text
credit-risk-scoring-service/
├── src/riskscore/          # 运行时包（纯标准库）
│   ├── config.py           # Settings、环境变量解析
│   ├── errors.py           # 错误层级（带 HTTP 状态码与稳定错误码）
│   ├── features.py         # 校验、变换、特征契约
│   ├── model.py            # 逻辑回归、梯度下降、指标、持久化
│   ├── scorecard.py        # PD → 分数/档位/原因码
│   ├── storage.py          # SQLite schema、ApplicationStore
│   ├── audit.py            # 只追加审计
│   ├── metrics.py          # 计数器、延迟直方图
│   ├── api.py              # 与传输层无关的路由
│   └── server.py           # http.server 传输层
├── scripts/
│   ├── generate_dataset.py # 合成数据生成器（固定种子）
│   ├── train_model.py      # 训练 → 校准 → 选阈值 → 出报告
│   ├── smoke_check.py      # 临时目录中的端到端自检
│   ├── bench.py            # 性能基准
│   └── diagnose.py         # 分数/档位排查工具
├── tests/                  # unittest 测试套件（244 个用例）
├── docs/DESIGN.md  ·  MODEL_CARD.md  ·  CONTRIBUTING.md
├── Makefile  ·  run.ps1    # 等价的两套命令入口
└── Dockerfile  ·  docker-compose.yml  ·  .github/workflows/ci.yml
```

## 快速开始

不需要虚拟环境，不需要 `pip install` 任何东西——只要有 Python 3.9+。

```bash
git clone https://github.com/Nausicaa613/credit-risk-scoring-service.git
cd credit-risk-scoring-service
```

### 使用 make

```bash
make data     # 生成合成数据（固定种子，结果可复现）
make train    # 训练、校准、选阈值，产出模型与报告
make test     # 244 个单元与集成测试
make smoke    # 端到端自检（临时目录，不污染工作区）
make run      # 启动服务，默认 http://127.0.0.1:8080
make bench    # 性能基准
```

### Windows 上没有 make

`run.ps1` 提供完全相同的目标名与默认参数：

```powershell
.\run.ps1 data
.\run.ps1 train
.\run.ps1 test
.\run.ps1 smoke
.\run.ps1 run
```

### 不借助任何构建工具

```bash
python scripts/generate_dataset.py --rows 6000 --seed 20260101 --output data/applications.jsonl
python scripts/train_model.py --data data/applications.jsonl --model models/model.json
python -m unittest discover -s tests -t . -v
.\run.ps1 run                       # 或 make run，两者都会自动设置 PYTHONPATH=src
python -m riskscore.server          # 仅在 make install（pip install -e .）之后可用
```

`scripts/` 下的脚本与 `conftest.py` 会自行把 `src/` 加入导入路径，`make run` / `run.ps1 run` 还会同时设好
`RISKSCORE_*` 环境变量与 `PYTHONPATH=src`，所以新克隆的仓库无需任何安装步骤即可启动。
只有希望 `riskscore` 在机器任意位置都可导入时，才需要执行 `make install`。

### 调一次分

```bash
curl -s -X POST http://127.0.0.1:8080/v1/score \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 40, "annual_income": 300000, "employment_years": 8.0,
    "debt_to_income_ratio": 0.30, "num_delinquencies_24m": 0,
    "credit_history_months": 150, "loan_amount": 200000,
    "loan_term_months": 36, "num_open_accounts": 5,
    "revolving_utilization": 0.30, "purpose": "equipment"
  }'
```

响应（字段名与实际一致，具体数值取决于你训练出的模型）：

```json
{
  "request_id": "req_5f3a91c2b7d4",
  "created_at": "2026-10-04T12:00:00.000Z",
  "result": {
    "application_id": "app_9c1f8e2a4b6d0f31",
    "credit_score": 690,
    "probability_of_default": 0.1472,
    "risk_band": "D",
    "risk_band_label": "elevated risk",
    "decision": "approve",
    "decision_threshold": 0.315,
    "model_version": "0.1.0",
    "expected_default_rate": "12% - 25%",
    "log_odds": -1.755,
    "intercept": -1.318,
    "contributions": [
      {"feature": "revolving_utilization", "contribution": -0.154},
      {"feature": "debt_to_income_ratio", "contribution": -0.109}
    ],
    "reason_codes": [
      {
        "feature": "revolving_utilization",
        "description": "revolving utilisation",
        "contribution": -0.154,
        "direction": "decreases_risk",
        "expected_direction": "risk_up",
        "consistent_with_prior": false
      }
    ]
  }
}
```

注意 `contributions` 与 `reason_codes` 都在响应里：一次拒绝如果只列出抬高风险的因素，审查者是没法复核的，所以两个方向都会给出。

## 实测结果

数据是**合成的**，下列指标只描述模型在这个合成分布上的表现，**不能**作为真实信贷数据上效果的证据。

```bash
make data && make train
```

| 项目 | 值 |
| --- | --- |
| 数据集 | 6000 行，违约率 0.2187（1312/6000），12% 标签噪声 |
| 划分 | 训练/验证/校准/测试 = 3600 / 600 / 600 / 1200 |
| 模型选择 | 独立验证集（早停于验证集 AUC） |
| 训练轮数 | 400（学习率 0.35，L2 0.001，种子 20260101） |

**留出测试集指标（阈值 0.315）**

| 指标 | 值 |
| --- | --- |
| AUC | 0.7337 |
| KS | 0.3934 |
| 准确率 | 0.7450 |
| 精确率 | 0.4380 |
| 召回率 | 0.5779 |
| F1 | 0.4984 |
| 预测为正例的比例 | 0.2892 |

混淆矩阵（行=实际，列=预测）：TN 742 ／ FP 195 ／ FN 111 ／ TP 152。

**概率校准**：类别加权训练后，平均预测 PD 为 0.4639，远高于校准集实际违约率 0.2283；拟合出的偏置为 **−1.0730**，修正后测试集平均预测 PD 为 **0.2541**，实际违约率 0.2192。

**贡献度离散度最高的特征**（在测试集上）

| 排名 | 特征 | 权重 | 贡献度标准差 |
| --- | --- | --- | --- |
| 1 | `revolving_utilization` | +0.3676 | 0.3789 |
| 2 | `debt_to_income_ratio` | +0.3650 | 0.3665 |
| 3 | `num_delinquencies_24m` | +0.1831 | 0.1821 |
| 4 | `log_annual_income` | −0.1621 | 0.1690 |
| 5 | `age` | −0.1611 | 0.1603 |

方向符合业务先验：循环额度使用率、负债收入比、逾期次数抬高风险；收入、年龄降低风险。

**已知局限**：分箱校准表显示中间区间仍系统性高估（第 2–4 箱偏差 +0.066 到 +0.100）。一个全局偏置只能对齐均值，无法修正由标签噪声带来的形状偏差。另外由于分档边界由固定 PD 切点算出，本数据集上 A 档几乎为空。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/score` | 对一份申请打分，返回 201 与 `Location` |
| `GET` | `/v1/applications/<id>` | 读取已存决策（原始入参与完整响应） |
| `GET` | `/v1/applications?limit=&offset=` | 最近的决策列表（limit 1–200，默认 20） |
| `GET` | `/v1/audit?limit=&application_id=` | 读取审计轨迹（limit 1–200，默认 20） |
| `GET` | `/v1/bands` | 风险分档定义 |
| `GET` | `/healthz` | 200 健康 / 503 降级（含模型与数据库状态） |
| `GET` | `/metrics` | 进程内指标（可配置关闭） |
| `GET` | `/` 或 `/v1` | 服务索引与端点清单 |

### 请求字段

11 个必填字段：`age`、`annual_income`、`employment_years`、`debt_to_income_ratio`、`num_delinquencies_24m`、`credit_history_months`、`loan_amount`、`loan_term_months`、`num_open_accounts`、`revolving_utilization`、`purpose`。

`purpose` 取值：`working_capital`、`equipment`、`expansion`、`inventory`、`refinance`、`personal`、`education`、`other`。

接受别名：`income`/`annualIncome` → `annual_income`，`dti` → `debt_to_income_ratio`，`loanAmount` → `loan_amount`，`delinquencies` → `num_delinquencies_24m`，`purpose_code` → `purpose`。可选的 `application_id` 由客户端提供时必须形如 6–64 位字母数字。

校验会**一次列出所有问题**，而不是遇到第一个就返回，客户端一次就能改完：

```json
{
  "error": {
    "code": "validation_error",
    "message": "payload failed schema validation",
    "details": {
      "problems": [
        {"field": "age", "problem": "is required"},
        {"field": "purpose", "problem": "is required"}
      ],
      "allowed_fields": ["age", "annual_income", "..."]
    }
  },
  "request_id": "req_5f3a91c2b7d4"
}
```

### 状态码

| 状态 | 错误码 | 含义 |
| --- | --- | --- |
| 400 | `validation_error` | 字段缺失、类型错误、越界、JSON 语法错误 |
| 404 | `not_found` | 路径或申请 ID 不存在 |
| 405 | `method_not_allowed` | 路径存在但不支持该方法（带 `Allow` 头） |
| 411 | `length_required` | 使用了 chunked 传输 |
| 413 | `payload_too_large` | 请求体超过上限（不缓冲；读尽后关闭连接） |
| 422 | `unprocessable_entity` | JSON 合法但语义不可行 |
| 503 | `model_unavailable` | 模型未加载（先执行 `make train`） |

## 配置

全部通过环境变量，默认值开箱即用。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RISKSCORE_HOST` | `127.0.0.1` | 监听地址 |
| `RISKSCORE_PORT` | `8080` | 监听端口 |
| `RISKSCORE_DB_PATH` | `data/riskscore.db` | SQLite 文件路径 |
| `RISKSCORE_MODEL_PATH` | `models/model.json` | 模型产物路径 |
| `RISKSCORE_DECISION_THRESHOLD` | `0.35` | 拒绝阈值（服务启动时生效） |
| `RISKSCORE_MAX_BODY_BYTES` | `65536` | 请求体上限 |
| `RISKSCORE_LOG_LEVEL` | `INFO` | 日志级别 |
| `RISKSCORE_METRICS_ENABLED` | `1` | 是否暴露 `/metrics` |

## 测试

```bash
make test          # 详细输出
make test-quiet    # 仅摘要
make check         # 编译检查 + 端到端自检
```

244 个用例，约 16 秒。测试用标准库 `unittest`，覆盖：字段校验与别名、变换正确性、AUC/KS 的排名与并列情形、训练收敛与确定性、贡献度精确可加、分数单调性、档位边界与标签一致性、SQLite 事务原子性、审计只追加语义、请求路由与全部错误码、以及**真实 socket** 上的 HTTP/1.1 keep-alive 分帧。

## 设计取舍

| 决策 | 备选方案 | 原因 |
| --- | --- | --- |
| 纯标准库 | FastAPI + scikit-learn | 零依赖意味着任何装了 Python 的机器都能跑，且没有需要审计和打补丁的依赖面 |
| 标准库 `http.server` | Flask / FastAPI | 请求生命周期完全显式，这正是评审会追问的部分；代价是没有异步与 WebSocket |
| SQLite | PostgreSQL | 单文件、零运维、可提交进测试；v0.1 的写入量根本用不到服务端数据库 |
| 线性评分卡 | 梯度提升 | 可解释性是模型的性质而非事后近似；牺牲部分 AUC 换取可审计 |
| JSON 模型产物 | pickle / joblib | 可 diff、可 code review、无任意代码执行风险；代价是加载时要校验格式版本 |
| 合成数据 | 德国信用 / Lending Club | 无授权限制、可逐字节复现、且**知道真实生成过程**，能区分"模型不行"和"数据没有信号" |
| 决策行与审计行同事务 | 两次独立写入 | 一次 fsync 而非两次（实测请求耗时从 59ms 降到 33ms），且不会出现"有决策没审计"的窗口 |
| 每次操作一个连接 | 连接池 / 单写线程 | 规避整类线程问题，代价是每次约 9ms 的连接开销；这是文档化的下一步优化 |

## 性能

在 Windows / Python 3.12 上的实测（`make bench`）：

| 操作 | 耗时 |
| --- | --- |
| 特征工程 | 约 13 µs / 行（18 维） |
| 训练 | 约 3 ms / epoch（1200 行） |
| 纯打分 `Scorer.score_vector` | 约 0.011 ms / 行 |
| 模型产物 | 3.8 KiB JSON，加载约 2.3 ms |
| 完整请求（含两次落库） | 约 33 ms，p50 27 ms |

打分本身是微秒级；请求耗时几乎全被 SQLite 的持久化提交占据。实测一次事务批量插入 100 行只需 25 ms（0.26 ms/行），这说明后续优化的方向是连接池、批量提交，或改用 `synchronous=NORMAL`（WAL 模式下这是常见做法，应用崩溃仍安全，只在地断电时可能丢失最近若干次提交）。

## 路线图

- 在同一接口下替换为梯度提升后端，保留 `contributions`（用 SHAP 值），以便对比 AUC 与可解释性的取舍。
- 公平性报告：按收入分位、贷款用途等切片做差异化影响分析。目前**没有**做任何公平性评估，这是一个明确的缺口。
- 认证与多租户，以及请求级配额。
- 存储驱动抽象 + PostgreSQL 实现；连接池以消除每次请求的连接开销。
- 漂移监控：分数分布与特征分布的在线对比。
- 把 `/metrics` 换成 Prometheus 文本格式，接入标准监控栈。

## 许可

MIT，见 [LICENSE](LICENSE)。

> **重要**：本项目是作品集与教学项目。数据是 `scripts/generate_dataset.py` 用固定种子生成的**合成数据**，不对应任何真实申请人。它**不是**信贷决策系统，**不得**用于对真实个人或企业做出授信决定。模型的预期用途、局限与已知缺口见 [MODEL_CARD.md](MODEL_CARD.md)。
