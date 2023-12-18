"""
model is downloaded in process, you must request access to Llama-2
code must be executed with a GPU available --> test on ec2
"""
import huggingface_hub
from transformers import AutoTokenizer
import transformers
import torch
import os


# login to huggingface --> set your token as environment variable
huggingface_hub.login(
    token=os.environ.get('huggingfacetoken'),
    add_to_git_credential=False,
    new_session=False,
    write_permission=True
)

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto"
)

sequences = pipeline(
    "What should I eat today? I'm a vegetarian who loves protein.",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

print('debug')
