import verifiers as vf
from transformers import trainer
from verifiers.prompts import few_shots
from verifiers.tools import python
from verifiers.utils import load_example_dataset

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.

Example:
<think>
Let's submit the answer.
</think>
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = load_example_dataset("math", split="train")
print(f"length of dataset is {len(dataset)}")

vf_env = vf.ToolEnv(
    dataset=dataset, system_prompt=TOOL_PROMPT, few_shot=[], tools=[python], max_steps=3
)
print(vf_env.system_prompt)

model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
model, tokenizer = vf.get_model_and_tokenizer(
    model_name, model_kwargs={"attn_implementation": "sdpa"}
)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()
print(f"run_name is {run_name}")

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_iterations = 2
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 2
# training_args.vllm_server_host = "host_name"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
