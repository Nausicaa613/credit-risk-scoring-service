"""A single-page demo UI for the scoring service, served at ``GET /app``.

Why this is not in ``api.py``
----------------------------
``api.py`` is deliberately transport-neutral: it turns a request into a
``Response`` and knows nothing about HTTP. An HTML page is a presentation
concern, so it is served by the HTTP transport instead, and it is *not* part of
the JSON API contract -- the endpoint list at ``GET /`` still contains only JSON
routes.

Why the markup is a Python string
---------------------------------
Keeping the page inside the package means there is no package-data
configuration, no path resolution and no way for the container image to ship
without it. The page is self-contained by design: no CDN, no build step and no
framework, so ``GET /app`` works on a machine with no network access.

Why it is generated rather than hand-written
--------------------------------------------
The form is built from :data:`~riskscore.features.NUMERIC_FIELDS`, and
``tests/test_webapp.py`` asserts that the label and step tables cover exactly
that schema. Add a feature to the model and the UI gains an input for it -- or
the test suite says so.
"""

from __future__ import annotations

import html
import json
from functools import lru_cache
from typing import Dict, List

from .features import NUMERIC_FIELDS, PURPOSE_CODES

#: Chinese labels for the form. The API stays English; the page is for a
#: Chinese-speaking reviewer, so only the presentation layer is translated.
FIELD_LABELS: Dict[str, str] = {
    "age": "年龄",
    "annual_income": "年收入",
    "employment_years": "在职年限",
    "debt_to_income_ratio": "负债收入比",
    "num_delinquencies_24m": "近 24 个月逾期次数",
    "credit_history_months": "信用历史月数",
    "loan_amount": "申请金额",
    "loan_term_months": "申请期限（月）",
    "num_open_accounts": "未结清账户数",
    "revolving_utilization": "循环额度使用率",
}

#: Input step per field: ratios move in hundredths, counts and money in whole
#: units. Kept separate from ``NUMERIC_FIELDS`` because it is a UI concern.
FIELD_STEPS: Dict[str, str] = {
    "age": "1",
    "annual_income": "1000",
    "employment_years": "1",
    "debt_to_income_ratio": "0.01",
    "num_delinquencies_24m": "1",
    "credit_history_months": "1",
    "loan_amount": "1000",
    "loan_term_months": "1",
    "num_open_accounts": "1",
    "revolving_utilization": "0.01",
}

#: A deliberately unremarkable applicant, used to prefill the form so the page
#: is useful on first load -- and to show the equivalent ``curl`` command.
SAMPLE_APPLICATION: Dict[str, object] = {
    "age": 35,
    "annual_income": 240000,
    "employment_years": 6,
    "debt_to_income_ratio": 0.32,
    "num_delinquencies_24m": 0,
    "credit_history_months": 96,
    "loan_amount": 150000,
    "loan_term_months": 36,
    "num_open_accounts": 5,
    "revolving_utilization": 0.38,
    "purpose": "working_capital",
}


def _number(value: float) -> str:
    """Render a bound without a pointless trailing ``.0``."""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _numeric_inputs() -> str:
    """Build one labelled number input per numeric feature, in schema order.

    The values are all local schema constants, never user input, so there is
    nothing here to escape.
    """
    lines: List[str] = []
    for field, (low, high, _description) in NUMERIC_FIELDS.items():
        lines.append(
            f'        <label class="field">\n'
            f'          <span>{FIELD_LABELS[field]} <code>{field}</code></span>\n'
            f'          <input type="number" name="{field}" value="{SAMPLE_APPLICATION[field]}" '
            f'min="{_number(low)}" max="{_number(high)}" step="{FIELD_STEPS[field]}" required>\n'
            f'        </label>'
        )
    return "\n".join(lines)


def _purpose_options() -> str:
    lines: List[str] = []
    for code in PURPOSE_CODES:
        selected = " selected" if code == SAMPLE_APPLICATION["purpose"] else ""
        lines.append(f'            <option value="{code}"{selected}>{code}</option>')
    return "\n".join(lines)


def _curl_example() -> str:
    """A copy-pasteable one-liner, generated from the same sample payload."""
    payload = json.dumps(SAMPLE_APPLICATION, ensure_ascii=False)
    return html.escape(
        "curl -s -X POST http://127.0.0.1:8080/v1/score \\\n"
        "     -H 'Content-Type: application/json' \\\n"
        f"     -d '{payload}'"
    )


_TEMPLATE = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>信用风险评分服务 · 演示</title>
<style>
  :root { --accent:#1f3a5f; --muted:#5b6472; --line:#dde3ec; --risk:#c0392b; --safe:#1e8449; }
  * { box-sizing: border-box; }
  body { margin:0; padding:0 20px 48px; font:14px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif; color:#20242b; background:#f6f8fb; }
  header { max-width:1100px; margin:0 auto; padding:28px 0 8px; }
  h1 { margin:0 0 4px; font-size:22px; color:var(--accent); }
  .sub { margin:0; color:var(--muted); }
  main { max-width:1100px; margin:0 auto; display:grid; grid-template-columns:minmax(0,380px) minmax(0,1fr); gap:18px; align-items:start; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .panel { background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  .panel h2 { margin:0 0 12px; font-size:15px; color:var(--accent); border-bottom:1px solid var(--line); padding-bottom:8px; }
  .field { display:block; margin-bottom:10px; }
  .field span { display:block; font-size:12px; color:var(--muted); }
  .field code { color:#8a94a6; font-size:11px; }
  .field input, .field select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; font:13px inherit; color:#20242b; background:#fff; }
  .field input:focus, .field select:focus { outline:2px solid #b9cbe4; border-color:var(--accent); }
  .actions { display:flex; gap:8px; margin-top:14px; }
  button { padding:8px 16px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:#fff; font:13px inherit; cursor:pointer; }
  button.secondary { background:#fff; color:var(--accent); }
  button:disabled { opacity:.55; cursor:progress; }
  .hint { min-height:20px; margin:10px 0 0; font-size:12px; color:var(--muted); }
  .hint.error { color:var(--risk); }
  .placeholder { color:var(--muted); }
  .scoreline { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  .score { font-size:44px; font-weight:700; color:var(--accent); line-height:1; }
  .badge { padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; background:#eef2f8; color:var(--accent); }
  .badge.approve { background:#e6f4ea; color:var(--safe); }
  .badge.decline { background:#fdecea; color:var(--risk); }
  .badge.review { background:#fff4e5; color:#b26a00; }
  dl.meta { display:grid; grid-template-columns:auto 1fr; gap:2px 14px; margin:16px 0 0; font-size:13px; }
  dl.meta dt { color:var(--muted); }
  dl.meta dd { margin:0; }
  h3 { font-size:13px; margin:20px 0 8px; color:var(--accent); }
  .contrib { display:grid; grid-template-columns:190px 1fr 72px; gap:6px 10px; align-items:center; font-size:12px; }
  .bar { position:relative; height:14px; background:#eef2f8; border-radius:3px; overflow:hidden; }
  .bar i { position:absolute; top:0; bottom:0; left:50%; background:var(--risk); border-radius:2px; }
  .bar i.down { background:var(--safe); }
  .contrib .val { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
  ul.reasons { margin:0; padding-left:18px; font-size:13px; }
  ul.reasons li { margin-bottom:4px; }
  .up { color:var(--risk); } .down { color:var(--safe); } .flat { color:var(--muted); }
  details { margin-top:18px; } summary { cursor:pointer; color:var(--accent); font-size:13px; }
  pre { background:#f2f5f9; border:1px solid var(--line); border-radius:6px; padding:10px; overflow:auto; font-size:12px; max-height:320px; }
  table { border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }
  th, td { border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; }
  th { color:var(--muted); font-weight:600; }
  footer { max-width:1100px; margin:24px auto 0; color:var(--muted); font-size:12px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>信用风险评分服务</h1>
  <p class="sub">模型接入 → 推理服务 → 用户交互的完整链路演示。本页只用浏览器原生能力：无框架、无 CDN、无构建步骤，全部数据来自公开的 JSON API。</p>
</header>
<main>
  <section class="panel">
    <h2>申请人信息</h2>
    <form id="score-form">
<!--NUMERIC_FIELDS-->
      <label class="field">
        <span>贷款用途 <code>purpose</code></span>
        <select name="purpose">
<!--PURPOSE_OPTIONS-->
        </select>
      </label>
      <div class="actions">
        <button type="submit" id="submit">打分</button>
        <button type="button" class="secondary" id="reset">填充示例</button>
      </div>
      <p class="hint" id="status"></p>
    </form>
  </section>

  <section class="panel">
    <h2>评分结果</h2>
    <div id="result" class="placeholder">填写左侧表单后点击「打分」。结果包含分数、档位、决策、每个特征的贡献值，以及原因码。</div>
    <details id="bands-details" hidden>
      <summary>档位定义（来自 GET /v1/bands）</summary>
      <div id="bands"></div>
    </details>
    <details>
      <summary>这个页面调用了什么？</summary>
      <p>页面只调用两个公开端点，因此它同时是接口契约的活文档：</p>
      <pre>POST /v1/score     — 打分并落库（返回 201 + Location）
GET  /v1/bands     — 档位定义</pre>
      <p>等价命令行：</p>
      <pre><!--CURL_EXAMPLE--></pre>
    </details>
  </section>
</main>
<footer>
  本页由 <code>GET /app</code> 提供，不属于 JSON API 契约；接口索引见 <a href="/">GET /</a>，健康检查见 <a href="/healthz">/healthz</a>。
</footer>

<script>
(function () {
  "use strict";

  var form = document.getElementById("score-form");
  var resultBox = document.getElementById("result");
  var statusLine = document.getElementById("status");
  var bandsDetails = document.getElementById("bands-details");
  var bandsBox = document.getElementById("bands");
  var submitButton = document.getElementById("submit");

  var DECISIONS = { approve: "通过", decline: "拒绝", review: "人工复核" };
  var DIRECTIONS = { increases_risk: "抬高风险", decreases_risk: "降低风险" };

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) { element.className = className; }
    if (text !== undefined) { element.textContent = text; }
    return element;
  }

  function payloadFromForm() {
    var payload = {};
    Array.prototype.forEach.call(form.elements, function (element) {
      if (!element.name) { return; }
      payload[element.name] = element.name === "purpose"
        ? element.value
        : Number(element.value);
    });
    return payload;
  }

  function setStatus(message, isError) {
    statusLine.textContent = message || "";
    statusLine.className = isError ? "hint error" : "hint";
  }

  function describeError(body, status) {
    var error = (body && body.error) || {};
    var problems = (error.details && error.details.problems) || [];
    var suffix = problems.length
      ? " — " + problems.map(function (p) { return p.field + " " + p.problem; }).join("；")
      : "";
    return (error.message || ("HTTP " + status)) + suffix;
  }

  function requestScore(payload) {
    return fetch("/v1/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.text().then(function (text) {
        var body;
        try { body = JSON.parse(text); }
        catch (parseError) { throw new Error("响应不是合法 JSON：" + text.slice(0, 160)); }
        if (!response.ok) { throw new Error(describeError(body, response.status)); }
        return body;
      });
    });
  }

  function renderContributions(result) {
    var wrapper = node("div", "contrib");
    var values = result.contributions || [];
    var largest = values.reduce(function (max, item) {
      return Math.max(max, Math.abs(item.contribution));
    }, 0) || 1;

    values.slice().sort(function (a, b) {
      return Math.abs(b.contribution) - Math.abs(a.contribution);
    }).forEach(function (item) {
      var width = (Math.abs(item.contribution) / largest) * 50;
      var positive = item.contribution >= 0;
      wrapper.appendChild(node("div", "", item.feature));
      var bar = node("div", "bar");
      var fill = node("i", positive ? "" : "down");
      fill.style.width = width + "%";
      fill.style.left = positive ? "50%" : (50 - width) + "%";
      bar.appendChild(fill);
      wrapper.appendChild(bar);
      wrapper.appendChild(node("div", "val", item.contribution.toFixed(4)));
    });
    return wrapper;
  }

  function renderReasons(result) {
    var list = node("ul", "reasons");
    (result.reason_codes || []).forEach(function (reason) {
      var direction = reason.direction === "increases_risk" ? "up" : "down";
      var item = node("li");
      item.appendChild(node("span", direction, DIRECTIONS[reason.direction] || reason.direction));
      item.appendChild(document.createTextNode(
        " " + reason.description + "（" + reason.feature + "，贡献 " +
        reason.contribution.toFixed(4) + "）"
      ));
      if (!reason.consistent_with_prior) {
        item.appendChild(node("strong", "up", " 与业务先验方向不一致"));
      }
      list.appendChild(item);
    });
    return list;
  }

  function renderResult(body) {
    var result = body.result;
    resultBox.className = "";
    resultBox.textContent = "";

    var line = node("div", "scoreline");
    line.appendChild(node("div", "score", String(result.credit_score)));
    line.appendChild(node("span", "badge", "档位 " + result.risk_band));
    line.appendChild(node("span", "badge " + result.decision, DECISIONS[result.decision] || result.decision));
    resultBox.appendChild(line);

    var meta = node("dl", "meta");
    [
      ["违约概率 PD", (result.probability_of_default * 100).toFixed(2) + "%"],
      ["决策阈值", result.decision_threshold],
      ["档位说明", result.risk_band_label],
      ["该档位经验违约率", result.expected_default_rate],
      ["模型版本", result.model_version],
      ["申请编号", result.application_id],
      ["log-odds / 截距", result.log_odds + " / " + result.intercept]
    ].forEach(function (pair) {
      meta.appendChild(node("dt", "", pair[0]));
      meta.appendChild(node("dd", "", String(pair[1])));
    });
    resultBox.appendChild(meta);

    resultBox.appendChild(node("h3", "", "特征贡献（log-odds，红=抬高风险，绿=降低风险）"));
    resultBox.appendChild(renderContributions(result));

    resultBox.appendChild(node("h3", "", "原因码"));
    resultBox.appendChild(renderReasons(result));

    var details = node("details");
    details.appendChild(node("summary", "", "原始响应 JSON"));
    details.appendChild(node("pre", "", JSON.stringify(body, null, 2)));
    resultBox.appendChild(details);
  }

  function loadBands() {
    fetch("/v1/bands").then(function (response) { return response.json(); }).then(function (body) {
      var table = node("table");
      var head = node("tr");
      ["档位", "说明", "分数下界（不含）", "分数上界（含）", "经验违约率"].forEach(function (title) {
        head.appendChild(node("th", "", title));
      });
      table.appendChild(head);
      (body.bands || []).forEach(function (band) {
        var row = node("tr");
        [band.band, band.label, band.lower_exclusive, band.upper_inclusive, band.expected_default_rate]
          .forEach(function (value) {
            row.appendChild(node("td", "", value === null || value === undefined ? "—" : String(value)));
          });
        table.appendChild(row);
      });
      bandsBox.textContent = "";
      bandsBox.appendChild(table);
      bandsDetails.hidden = false;
    }).catch(function () {
      /* The band table is decoration; a failure here must not affect scoring. */
    });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitButton.disabled = true;
    setStatus("打分中…", false);
    requestScore(payloadFromForm())
      .then(function (body) {
        renderResult(body);
        setStatus("完成：申请编号 " + body.result.application_id, false);
      })
      .catch(function (error) {
        setStatus(error.message, true);
      })
      .then(function () {
        submitButton.disabled = false;
      });
  });

  document.getElementById("reset").addEventListener("click", function () {
    form.reset();
    setStatus("已恢复示例数据", false);
  });

  loadBands();
})();
</script>
</body>
</html>
'''


@lru_cache(maxsize=1)
def render_page() -> str:
    """Return the demo page with the form generated from the feature schema.

    Cached because the page is a pure function of the code: the schema cannot
    change while the process is running, so rebuilding the markup on every
    request would be wasted work.
    """
    return (
        _TEMPLATE
        .replace("<!--NUMERIC_FIELDS-->", _numeric_inputs())
        .replace("<!--PURPOSE_OPTIONS-->", _purpose_options())
        .replace("<!--CURL_EXAMPLE-->", _curl_example())
    )
