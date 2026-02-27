FROM nvcr.io/nvidia/tritonserver:23.12-py3

# Install PyTorch and Torchvision
# We need to ensure compatibility with the CUDA version in the base image (usually 12.x for 23.12)
# Using pip install with NVIDIA index
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
