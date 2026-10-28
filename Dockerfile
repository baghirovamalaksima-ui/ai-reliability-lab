FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY core ./core
RUN pip install --no-cache-dir '.[test]' \
    && useradd --create-home --uid 10001 evaluator
COPY tests ./tests
USER evaluator
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
