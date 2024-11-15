# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    cmake \
    libopencv-dev \
    libopencv-core-dev && \
    rm -rf /var/lib/apt/lists/*

# Create workspace directory
RUN mkdir /workspace

# Set the working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/YWHyuk/simple-cpu-MLP.git

# Move to the project directory
WORKDIR /workspace/simple-cpu-MLP

# Create and move to the build directory
RUN mkdir build && cd build

# Run cmake and make commands
RUN cd build && cmake .. && make