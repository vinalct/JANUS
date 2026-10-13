FROM python:3.15.0rc3-slim-bookworm@sha256:ea93de1a70a03fffd10ad8cfde8f4ba969fe2658f0578af02855d1d32f49e6c2

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    JANUS_SYSTEM_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    HOME=/tmp

WORKDIR /workspace

COPY conf/certs/rnpicpedugr46ovtlsca2025.pem /usr/local/share/ca-certificates/rnpicpedugr46ovtlsca2025.crt

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates make openssl openjdk-17-jre-headless libnss-wrapper procps \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 janus \
    && useradd --uid 1000 --gid 1000 --home-dir /tmp --shell /usr/sbin/nologin janus

# Source changes should reuse the expensive PySpark and runtime dependency layer.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY conf ./conf
COPY .github ./.github
COPY docs ./docs
COPY examples ./examples
COPY tests ./tests
COPY --chown=1000:1000 data ./data

RUN python -m pip install --no-deps .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

USER 1000:1000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "janus.main", "--environment", "local", "--with-spark"]
