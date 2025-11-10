---
hide:
  - navigation
  - toc
---
# vLLM Fork for MXFP4 Quantization on Legacy ROCm GPUs

This repository enables the use of models with **MXFP4 quantization**, such as the [GPT-OSS](https://huggingface.co/collections/openai/gpt-oss) series, on older AMD Instinct GPUs.

It is based on the work in [`zyongye/vllm`](https://github.com/zyongye/vllm/tree/triton_kernels_legacy_routing), which fixed critical compatibility issues between newer Triton kernels and vLLM's Mixture-of-Experts routing code. This fork adds a small but essential fix on top to prevent an import error on pre-GFX950 hardware, which is necessary for MXFP4 support on these cards.

## Model & Hardware Compatibility

This fork has been successfully tested running the **20-billion parameter GPT-OSS model** on an **AMD Instinct MI100** GPU.

It is expected that the **120-billion parameter model** may also work, depending on available VRAM. The fix should also enable support for other GPUs from the same generation, such as the **MI60** and **MI210**, though these have not been tested.

## Installation

These instructions provide a complete, reproducible setup for a **ROCm 7.0.2** environment. All key dependencies are pinned to specific versions that are known to work together for this use case.

```bash
# 1. Create and activate a Python virtual environment.
# This isolates the project's dependencies from your system.
python3 -m venv venv && source venv/bin/activate

# 2. Install build prerequisites.
pip install --upgrade pip
pip install -U packaging 'cmake<4' ninja wheel setuptools pybind11 Cython
# Note: The path to amd_smi may vary depending on your ROCm installation.
pip install /opt/rocm/share/amd_smi

# 3. Install the specific ROCm PyTorch nightly build.
# These versions are pinned to ensure compatibility with the custom kernels.
pip3 install --no-cache-dir \
  torch==2.10.0.dev20251109+rocm7.0 \
  torchaudio==2.10.0.dev20251110+rocm7.0 \
  torchvision==0.25.0.dev20251110+rocm7.0 \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# 4. Install the required Triton Kernels library.
# The vLLM code for MXFP4 explicitly depends on this exact commit of the triton_kernels library.
# This is installed in addition to the Triton compiler bundled with PyTorch.
pip install "triton_kernels @ git+https://github.com/triton-lang/triton.git@c3c476f357f1e9768ea4e45aa5c17528449ab9ef#subdirectory=python/triton_kernels"

# 5. Clone and install this vLLM fork.
# This method ensures all dependencies from requirements-rocm.txt are installed correctly.
git clone https://github.com/dazipe/vllm.git
cd vllm
pip install -r requirements-rocm.txt
# Install the project in development mode.
python setup.py develop
```


# Welcome to vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

Where to get started with vLLM depends on the type of user. If you are looking to:

- Run open-source models on vLLM, we recommend starting with the [Quickstart Guide](./getting_started/quickstart.md)
- Build applications with vLLM, we recommend starting with the [User Guide](./usage)
- Build vLLM, we recommend starting with [Developer Guide](./contributing)

For information about the development of vLLM, see:

- [Roadmap](https://roadmap.vllm.ai)
- [Releases](https://github.com/vllm-project/vllm/releases)

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular HuggingFace models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

For more information, check out the following:

- [vLLM announcing blog post](https://vllm.ai) (intro to PagedAttention)
- [vLLM paper](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference) by Cade Daniel et al.
- [vLLM Meetups](community/meetups.md)
