# Build image for the optional `capture` feature (raw capture via libcamera).
#
# Why a container: the libcamera Rust bindings need libcamera-dev + clang at
# build time, which we do not want on the host. The image MUST be ubuntu:26.04
# so its libcamera matches the host runtime (libcamera.so.0.7) for ABI
# compatibility — the resulting binary links dynamically and runs on the host.
#
# --http-proxy=false: the host runs a proxy at 127.0.0.1, which podman would
# forward into the container, where 127.0.0.1 is not the proxy. The build/run
# network reaches the internet directly via NAT, so disable proxy forwarding.
#
# Build the image:
#   podman build --http-proxy=false -t gc2607-isp-build -f Containerfile .
# Build the capture binary (cargo cache mounted to avoid re-downloads):
#   podman run --rm --http-proxy=false -v "$PWD":/work \
#     -v gc2607-cargo-registry:/root/.cargo/registry \
#     -v gc2607-cargo-target:/work/target \
#     gc2607-isp-build \
#     cargo build --release --features capture --bin gc2607-capture
# The binary then runs on the host (libcamera runtime is present there).

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        build-essential pkg-config clang \
        libcamera-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (pinned to match the host: 1.88.0). HOME/CARGO_HOME/RUSTUP_HOME
# are set explicitly: `podman build` does not set HOME, so rustup would
# otherwise install outside /root and cargo would be missing from PATH.
ENV HOME=/root CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain 1.88.0 \
    && cargo --version

WORKDIR /work
