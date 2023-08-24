from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain import PromptTemplate

llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_K_S.bin")

template = "Who directed {movie_name}."
prompt = PromptTemplate.from_template(input_variable=["adjective"], template=template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run("The Dark Knight")
print(response)