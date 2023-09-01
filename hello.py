from llama_cpp import Llama

llm = Llama(model_path="./models/llama-7b.ggmlv3.q4_K_S.bin")
response = llm("Quem é você?")

print(response['choices'][0]['text'])