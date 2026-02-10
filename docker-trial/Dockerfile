FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install basic build tools and wget
RUN apt-get update && apt-get install -y time wget g++ libboost-all-dev git && \
    apt-get clean

# Install Miniforge (Conda) + Mamba
ENV CONTAINER_HOME=/opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \\
    /bin/bash /tmp/miniforge.sh -b -p ${CONTAINER_HOME}
ENV PATH=${CONTAINER_HOME}/bin:$PATH

# Create hyper-spec environment from requirements
COPY docker/requirements.yaml /tmp/requirements.yaml
RUN mamba env create --name hyper-spec -f /tmp/requirements.yaml

# Make sure the conda env is activated by default
RUN echo \"source activate hyper-spec\" >> ~/.bashrc

# Install Nextflow for the incremental workflow
RUN mamba install -y -c bioconda nextflow=23.* && mamba clean -afy

# Clone Hyper-Spec into the image (required by incremental workflow)
WORKDIR /tmp
RUN git clone https://github.com/wh-xu/Hyper-Spec.git && \\
    mv Hyper-Spec /Hyper-Spec

# Copy this Incremental_Clustering repository into the image
COPY . /Incremental_Clustering

# Build Hyper-Spec inside the container
WORKDIR /Hyper-Spec
RUN /bin/bash -c \"source activate hyper-spec && bash ./install.sh\"

# Default working directory for the incremental workflow
WORKDIR /Incremental_Clusterिंग

