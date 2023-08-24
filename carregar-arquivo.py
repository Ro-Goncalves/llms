from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import LlamaCppEmbeddings

#load Doc
loader = TextLoader("./docs/file.txt")
docs = loader.load()

#Transform into chunks
text_spliter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_spliter.split_documents(docs)

#compare size of docs and text
print("\n\nDocs size: "  + str(len(docs)))
print("Texts size: " + str(len(texts)) + '\n')

#visualize chunks and doc
print(str(docs) + "\n")
print(texts[0])

# convert doc to str
_texts = []
for i in range(len(texts)):
    _texts.append(texts[i].page_content)

# visualize
print(_texts[0])

# embed list of texts
embeddings = LlamaCppEmbeddings(model_path="./models/llama-7b.ggmlv3.q4_K_S.bin")
embedded_texts = embeddings.embed_documents(_texts)

print("\nLen embedded texts: " + str(len(embedded_texts)))
print("Len embedded texts: "   + str(len(embedded_texts[0])) + "\n")

# embed query
query = "What skills did batman had?"
embedded_query = embeddings.aembed_query(query)
#print("Len embedded query: "    + str(len(embedded_query)))
#print("Vector representation: " + str(embedded_query[:4]))