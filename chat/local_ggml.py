"""
download models that can run locally without GPU here:
https://huggingface.co/TheBloke/Llama-2-7B-GGUF
"""

from llama_cpp import Llama

LLM = Llama(model_path="../models/llama-2-7b.Q2_K.gguf")

# create a text prompt
prompt = "Q: What should I eat today? I'm a vegetarian who loves protein. A:"

# generate a response (takes several seconds)
output = LLM(prompt)

# display the response
print(output["choices"][0]["text"])
