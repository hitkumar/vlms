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
- verify that the model training works. Training is still not working as VLLM connection to produce generations is broken.
- For now grpo_demo.py works, lets use this to test the RL training. Will need to go deep into verifiers to understand how to get the vllm connection to work.

Open source RL libraries
- Blog: https://www.anyscale.com/blog/open-source-rl-libraries-for-llms
- TRL is good for single turn text based RL
- verifiers builds on top of TRL and adds support for multi-turn agentic RL. Will be interesting to extenbd to multimodal RL use-cases.
- Verl is good for large model training where performance is critical. It also has some variants with multiturn agenticb support like Ragen.
- New entrant SkyRL could be interesting to keep an eye on.


https://www.perplexity.ai/hub/blog/rl-training-for-math-reasoning
- Mentios using nemoRL which has since been deprecated.

RL track at AI Engineer
- Talk from Will covering verifiers: https://www.youtube.com/watch?v=PbHm2qKnu10&list=PLcfpQ4tk2k0V16VYYwnwF2g-EsKRIkJaC&index=3
