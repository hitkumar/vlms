Attempting to add reasoning ability to VLMs via verifiers library

- https://github.com/willccbb/verifiers

From the docs:
https://github.com/willccbb/verifiers/blob/main/docs/source/overview.md

there are 4 key components of the library:
- Environments
- Parsers
- Rubrics
- Training models via HF TRL

Video sessions on verifiers
- https://maven.com/p/c3950c/training-agents-with-reinforcement-learning

For running examples

- Start vllm server

CUDA_VISIBLE_DEVICES=0 python -m verifiers.inference.vllm_server --model 'willcb/Qwen2.5-7B-Math-Python-SFT' --max-model-len 8192 --dtype bfloat16     --gpu-memory-utilization 0.9 --enable-prefix-caching     --host $(hostname) --port 8000

- Start Training
CUDA_VISIBLE_DEVICES=1 accelerate launch --config-file rl/configs/zero3.yaml  rl/examples/math_python.py

Next steps
- verify that the model training works.
