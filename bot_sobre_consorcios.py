# Bring in deps
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# set prompt template
prompt_template = """Use as seguintes partes do contexto para responder à pergunta no final. Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.

{context}

Pergunta: {question}
Resposta:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# initialize the LLM & Embeddings
llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_K_S.bin")
embeddings = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q4_K_S.bin")
llm_chain = LLMChain(llm=llm, prompt=prompt)

#load Doc
loader = TextLoader("./docs/sobre-consorcio.txt")
docs = loader.load()
text_spliter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_spliter.split_documents(docs)
db = Chroma.from_documents(texts, embeddings)

question = "Encontra algo semelhante a: ...Como eu contemplo meu consorcio?... no texto?" 

similar_doc = db.similarity_search(question, k=1)
context = similar_doc[0].page_content
query_llm = LLMChain(llm=llm, prompt=prompt)
response = query_llm.run({"context": context, "question": question})        
print(response)
