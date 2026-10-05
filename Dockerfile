# 构建与运行镜像
# 两阶段构建：第一阶段只在需要时使用，本镜像无编译型依赖，因此直接基于 slim 镜像。
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="credit-risk-scoring-service" \
      org.opencontainers.image.description="A dependency-free credit risk scoring microservice" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="0.1.0"

# Python 运行参数：
#   PYTHONDONTWRITEBYTECODE 容器内不需要 .pyc，避免在只读层写入
#   PYTHONUNBUFFERED         日志立即输出，否则 docker logs 会延迟
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RISKSCORE_HOST=0.0.0.0 \
    RISKSCORE_PORT=8080 \
    RISKSCORE_DB_PATH=/app/data/riskscore.db \
    RISKSCORE_MODEL_PATH=/app/models/model.json

WORKDIR /app

# 先复制源码并安装，最大化利用层缓存：源码变更不会重复安装依赖。
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml README.md LICENSE /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 数据目录需要在运行时写入 SQLite 文件，因此显式创建并交给非 root 用户。
RUN mkdir -p /app/data /app/models /app/reports && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# 健康检查使用服务自身的 /healthz，模型缺失时返回 503，容器编排可以据此判断就绪。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,sys,urllib.request; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)); \
sys.exit(0 if r.get('status')=='ok' else 1)"

# 直接以模块方式启动，不依赖脚本文件的可执行位。
CMD ["python", "-m", "riskscore.server"]
