from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import openai

# 🔐 Tu clave de OpenAI
openai.api_key = "sk--"  # Reemplázala por la tuya

# 📄 Leer el documento
with open("data/documentos.txt", encoding="utf-8") as f:
    texto = f.read()

# 🧩 Fragmentar el texto
chunks = [Document(page_content=texto[i:i+500]) for i in range(0, len(texto), 500)]

# 🔍 Crear embeddings y FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 🔁 Construir la cadena de respuesta con retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 🎯 Función para responder
def responder(pregunta: str) -> str:
    try:
        return qa_chain.run(pregunta)
    except Exception as e:
        return f"[Error] {str(e)}"
# sfff