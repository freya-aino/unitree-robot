FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

RUN apt-get update -y && apt-get upgrade -y

RUN apt-get install -y --no-install-recommends curl ca-certificates

# install uv
ADD --chmod=007 https://astral.sh/uv/install.sh /root/uv-installer.sh
RUN sh /root/uv-installer.sh && rm /root/uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# # install project
WORKDIR /app/
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .
RUN uv venv /app/.venv --python="$(cat .python-version)"

# ENV VIRTUAL_ENV=/app/.venv
# ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN uv sync

ENTRYPOINT [ "/bin/bash" ]