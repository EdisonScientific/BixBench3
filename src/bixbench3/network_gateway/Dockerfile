FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git squid \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai==2.36.0 pydantic==2.13.4

WORKDIR /app
COPY src/bixbench3/network_gateway /app/bixbench3/network_gateway

RUN printf '%s\n' \
      '#!/usr/bin/env sh' \
      'exec python -m bixbench3.network_gateway.acl_helper' \
      > /usr/local/bin/bixbench3-acl-helper \
    && chmod +x /usr/local/bin/bixbench3-acl-helper

ENTRYPOINT ["python", "-m", "bixbench3.network_gateway"]
