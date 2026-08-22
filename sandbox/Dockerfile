ARG INSPECT_TOOL_SUPPORT_IMAGE=aisiuk/inspect-tool-support:latest
FROM ${INSPECT_TOOL_SUPPORT_IMAGE}

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        build-essential \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        gfortran \
        git \
        jq \
        libbz2-dev \
        libcairo2-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libfribidi-dev \
        libgit2-dev \
        libharfbuzz-dev \
        libhdf5-dev \
        libjpeg-dev \
        liblzma-dev \
        libpng-dev \
        libssl-dev \
        libtiff-dev \
        libxml2-dev \
        libzstd-dev \
        make \
        openjdk-17-jre-headless \
        pigz \
        pkg-config \
        python3 \
        python3-pip \
        r-base \
        r-base-dev \
        r-cran-data.table \
        r-cran-dplyr \
        r-cran-ggplot2 \
        r-cran-matrix \
        r-cran-readr \
        r-cran-remotes \
        r-cran-tidyr \
        rsync \
        tar \
        unzip \
        wget \
        zip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && ln -s /usr/local/bin/micromamba /usr/local/bin/mamba \
    && micromamba config append channels conda-forge \
    && micromamba config append channels bioconda \
    && micromamba config append channels defaults \
    && micromamba config set channel_priority flexible

RUN uv pip install --system \
        biopython \
        h5py \
        matplotlib \
        numpy \
        pandas \
        pyarrow \
        pysam \
        scikit-learn \
        scipy \
        statsmodels

RUN Rscript -e 'install.packages(c("BiocManager", "pak"), repos = "https://cloud.r-project.org")'

ARG CELLRANGER_TARBALL=operator-cache/10x/cellranger-6.0.2.tar.gz
ARG CELLRANGER_ARC_TARBALL=operator-cache/10x/cellranger-arc-2.0.2.tar.gz
ARG CELLRANGER_SHA256=""
ARG CELLRANGER_ARC_SHA256=""

COPY ${CELLRANGER_TARBALL} /tmp/cellranger-6.0.2.tar.gz
COPY ${CELLRANGER_ARC_TARBALL} /tmp/cellranger-arc-2.0.2.tar.gz

RUN if [ -n "${CELLRANGER_SHA256}" ]; then \
        echo "${CELLRANGER_SHA256}  /tmp/cellranger-6.0.2.tar.gz" | sha256sum -c -; \
    fi \
    && if [ -n "${CELLRANGER_ARC_SHA256}" ]; then \
        echo "${CELLRANGER_ARC_SHA256}  /tmp/cellranger-arc-2.0.2.tar.gz" | sha256sum -c -; \
    fi \
    && mkdir -p /opt/10x \
    && tar -xzf /tmp/cellranger-6.0.2.tar.gz -C /opt/10x \
    && tar -xzf /tmp/cellranger-arc-2.0.2.tar.gz -C /opt/10x \
    && test -x /opt/10x/cellranger-6.0.2/cellranger \
    && test -x /opt/10x/cellranger-arc-2.0.2/cellranger-arc \
    && ln -sf /opt/10x/cellranger-6.0.2/cellranger /usr/local/bin/cellranger \
    && ln -sf /opt/10x/cellranger-arc-2.0.2/cellranger-arc /usr/local/bin/cellranger-arc \
    && cellranger --version \
    && cellranger-arc --version \
    && rm -f /tmp/cellranger-6.0.2.tar.gz /tmp/cellranger-arc-2.0.2.tar.gz

WORKDIR /workspace
