FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 4000
# Client IP for rate limiting is derived in app.main.client_ip (TRUST_PROXY_HEADERS),
# not via --forwarded-allow-ips='*', which would let clients spoof X-Forwarded-For.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-4000}"]
