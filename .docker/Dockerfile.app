# 1. Base Image
# Use a slim Python image matching the project's version
FROM python:3.11-slim

# 2. Set Environment Variables
# Prevents Python from writing pyc files and buffers stdout and stderr
ENV WORKSPACE_ROOT=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/

RUN mkdir -p $WORKSPACE_ROOT

# 3. Install uv
# Install the uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/


ENV UV_HTTP_TIMEOUT=300 UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 4. Set Working Directory
WORKDIR $WORKSPACE_ROOT

# 5. Copy and Install Dependencies
# Copy dependency definition files first to leverage Docker cache
COPY pyproject.toml uv.lock ./
# Install dependencies using uv sync for a reproducible environment
RUN uv sync --locked

# 6. Copy Application Code
# Copy the rest of the application code
COPY ./app .

# 7. Expose Port
# Expose the default port for Gradio applications
EXPOSE 9011

# 8. Set Default Command
# Run the Gradio application, binding to 0.0.0.0 to make it accessible outside the container
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9011"]

