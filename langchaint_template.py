from langchain.llms import LlamaCpp
from langchain import PromptTemplate

llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_K_S.bin")

template = """Q: Who directed {movie_name}


Answer: """

prompt = PromptTemplate.from_template(template)

formatted_prompt = prompt.format(movie_name="The Dark Knight")

print(llm(prompt = formatted_prompt, stop=["Q:", "\n"]))