FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt

# Copy source code
COPY src/client/ ./client/
COPY src/core/ ./core/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
