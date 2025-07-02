FROM python:3.10-slim

COPY . /heimdall/attribute

ENV SPARSIFY_DISABLE_TRITON=1
ENV OFFLOAD_TRANSCODER=1
# ENV HF_TOKEN=$(cat /heimdall/attribute/.hf_auth)

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git
ADD https://astral.sh/uv/0.7.13/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

USER nobody
WORKDIR /heimdall/attribute
RUN uv venv && uv pip install -e . && uv lock