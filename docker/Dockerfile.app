FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
COPY requirements.txt .
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --no-dev
RUN uv pip install --system -r requirements.txt

COPY src/agents/ ./agents/
COPY src/core/ ./core/
COPY src/schema/ ./schema/
COPY src/client/ ./client/
COPY src/streamlit_app.py .

CMD ["streamlit", "run", "streamlit_app.py"]
