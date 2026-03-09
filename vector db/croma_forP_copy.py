import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"token": hf_token}
)



doc1 = Document(
    page_content="Python is one of the most popular programming languages for data science, artificial intelligence, and web development. Its simple syntax makes it beginner friendly.",
    metadata={"category": "AI & Data Science"}
)

doc2 = Document(
    page_content="Java is a widely used object-oriented programming language known for its portability. It is heavily used in enterprise software and Android development.",
    metadata={"category": "Enterprise Development"}
)

doc3 = Document(
    page_content="JavaScript is the backbone of web development. It allows developers to create interactive websites and works both in browsers and on servers using Node.js.",
    metadata={"category": "Web Development"}
)

doc4 = Document(
    page_content="C++ is a high-performance programming language used in system programming, game development, and applications requiring speed and memory control.",
    metadata={"category": "System Programming"}
)

doc5 = Document(
    page_content="Go, also known as Golang, was developed by Google and is widely used for cloud computing and backend services because of its efficiency and concurrency support.",
    metadata={"category": "Cloud & Backend"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="db1",
    collection_name="demo"
)

#add documents
vector_store.add_documents(docs)

#view stored documents
vector_store.get(include=["embeddings", "documents", "metadatas"])


#similarity search
vector_store.similarity_search(
    query="Which language is used for backend development?",
    k=2
)

vector_store.similarity_search_with_score(
    query="Which language is used for backend development?",
    k=2
)


#metadata filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"category": "Web Development"}
)


#update document
updated_doc1 = Document(
    page_content="Python is a versatile programming language widely used in artificial intelligence, machine learning, automation, and web development. Frameworks like TensorFlow, PyTorch, and Django make it extremely powerful.",
    metadata={"category": "AI & Data Science"}
)

vector_store.update_document(
    document_id="09a39dc6-3ba6-4ea7-927e-fdda591da5e4",
    document=updated_doc1
)


#view documents
result1 = vector_store.get(include=["embeddings", "documents", "metadatas"])
print(result1)


#delete document
vector_store.delete(ids=["09a39dc6-3ba6-4ea7-927e-fdda591da5e4"])


#view documents again
result2 = vector_store.get(include=["embeddings", "documents", "metadatas"])
print(result2)