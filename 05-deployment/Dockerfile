FROM agrigorev/zoomcamp-model:2025
RUN pip install uv
WORKDIR /app
COPY ["pyproject.toml", "uv.lock", "./"]
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync --frozen --no-dev
COPY ["predict-fastapi.py", "pipeline_v1.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["/app/.venv/bin/uvicorn", "predict-fastapi:app", "--host", "0.0.0.0", "--port", "9696"]