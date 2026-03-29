# Dockerfile for Summit-Sim Chainlit Application
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without syncing project)
RUN uv sync --all-extras --frozen

# Copy application code
COPY src/ ./src/
COPY .chainlit/ ./.chainlit/
COPY public/ ./public/
COPY chainlit.md ./
RUN ln -s chainlit.md chainlit_en-US.md

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Chainlit port
EXPOSE 8000

# Run Chainlit with hot reload for development
CMD ["uv", "run", "chainlit", "run", "src/summit_sim/app.py", "--host", "0.0.0.0", "--port", "8000"]
