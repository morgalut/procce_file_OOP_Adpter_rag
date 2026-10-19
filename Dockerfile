FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
# PyMuPDF and PostgreSQL client dependencies use wheels where available.
# gcc is included for Python dependencies that require a source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq5 && rm -rf /var/lib/apt/lists/*
COPY . /app
# Install project dependencies from the repository's source of truth.
RUN if [ -f pyproject.toml ]; then \
      python -m pip install --upgrade pip && python -m pip install .; \
    elif [ -f requirements.txt ]; then \
      python -m pip install --upgrade pip && python -m pip install -r requirements.txt; \
    else \
      echo 'ERROR: pyproject.toml or requirements.txt is required' >&2; exit 1; \
    fi
RUN mkdir -p /app/data/uploads && \
    useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app/data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "qbr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
