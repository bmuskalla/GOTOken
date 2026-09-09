# GOTOken: an LLM inference engine in BASIC (QB64), running SmolLM2-135M.
#
#   docker build -t gotoken .
#   docker run -it gotoken                                  # the REPL
#   docker run gotoken complete 32 "Once upon a time"       # one completion
#
# Three stages: export the model with Python, build the engine with QB64,
# then a slim runtime holding just the binary and the two model files.

# --- 1. export weights.bin + tokenizer.bin from the HuggingFace checkpoint ---
FROM python:3.11-slim AS export
RUN pip install --no-cache-dir numpy huggingface_hub tokenizers
WORKDIR /work
COPY export/export.py .
RUN python export.py --out /model

# --- 2. build QB64 from its release tarball, then compile the engine ---------
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ gcc curl ca-certificates mesa-common-dev libglu1-mesa-dev libasound2-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
ARG QB64_URL=https://github.com/QB64Official/qb64/releases/download/v2.1/qb64_dev_2022-09-08-07-14-00_47f5044_lnx.tar.gz
WORKDIR /opt
RUN curl -sSL -o qb64.tar.gz "$QB64_URL" && tar xzf qb64.tar.gz && mv qb64_*_lnx qb64 && rm qb64.tar.gz
WORKDIR /opt/qb64
# setup_lnx.sh minus the package manager and the IDE launch
RUN find . -name "*.sh" -exec chmod +x {} \; \
    && (cd internal/c/libqb/os/lnx && ./setup_build.sh) \
    && (cd internal/c/parts/video/font/ttf/os/lnx && ./setup_build.sh) \
    && (cd internal/c/parts/core/os/lnx && ./setup_build.sh) \
    && cp -r internal/source/* internal/temp/ \
    && (cd internal/c && g++ -no-pie -w qbx.cpp libqb/os/lnx/libqb_setup.o \
          parts/video/font/ttf/os/lnx/src.o parts/core/os/lnx/src.a \
          -lGL -lGLU -lX11 -lpthread -ldl -lrt -D FREEGLUT_STATIC -o ../../qb64)
WORKDIR /src
COPY gotoken.bas .
COPY src ./src
RUN /opt/qb64/qb64 -x -c /src/gotoken.bas -o /src/gotoken && ldd /src/gotoken

# --- 3. runtime: the binary and the model, nothing else ----------------------
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /src/gotoken /app/gotoken
COPY --from=export /model/weights.bin /model/tokenizer.bin /app/model/
# gotoken resolves model/ relative to the directory it is launched from
ENTRYPOINT ["/app/gotoken"]
CMD ["repl"]
