#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
#FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04


# Update and install mini-conda
RUN apt-get update && apt-get install -y time wget g++ libboost-all-dev git && \
        apt-get clean

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH


# setup conda virtual environment
COPY ./docker-trial/requirements.yaml /tmp/requirements.yaml
#RUN mamba create -n hyper-spec-new -c rapidsai -c nvidia -c conda-forge \
#    cuml=24.10 python=3.10 cuda-version=12.0

RUN mamba env create --name hyper-spec -f /tmp/requirements.yaml

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc
RUN echo "source activate hyper-spec" >> ~/.bashrc

#RUN /bin/bash -c "source activate hyper-spec && pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com ucx-py-cu12"

# RUN echo "conda activate hyper-spec" >> ~/.bashrc
# ENV PATH /opt/conda/envs/hyper-spec/bin:$PATH
# ENV CONDA_DEFAULT_ENV $hyper-spec

# Nvidia nvidia-utils
RUN apt-get update && apt-get install -y --no-install-recommends nvidia-utils-525

# Upgrade libstdc++6 to ensure compatibility with conda packages that require newer CXXABI
# This is needed for packages like libicui18n that require CXXABI_1.3.15
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 && \
    apt-get clean

# Install Nextflow for incremental clustering workflow
# Use version 22.x for compatibility with the workflow
RUN mamba install -y -c bioconda nextflow=23.*

# # Clone the code
# RUN git clone https://github.com/wh-xu/Hyper-Spec.git 

# Copy the code in
COPY ./docker-trial/Hyper-Spec /Hyper-Spec

# Copy this Incremental_Clustering repo into the image (build context root)
COPY . /Incremental_Clustering

# set working dir
WORKDIR /Hyper-Spec

# Install and build  
RUN /bin/bash -c "source activate hyper-spec && bash ./install.sh"

# copy over run_test.sh
COPY ./docker-trial/run_test.sh /Hyper-Spec/run_test.sh
