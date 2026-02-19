

# Create Vectore Embeddings 
# Store Embeddings in FAISS

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load raw PDFs
DATA_PATH="pdf_files/"
def load_pdf_files(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents=load_pdf_files(pdf_directory=DATA_PATH)
print("lenght of documents: ",len(documents))

#Step 2: Split documents into chunks
def create_chunks(extracted_data,chunk_size=500,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("lenght of text_chunks: ",len(text_chunks))

# Step 3: Create vector embeddings for the chunks
def get_embeddings_model():
    embeddings_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

embeddings_model=get_embeddings_model()

# # Step 4: Store the embeddings in FAISS
DB_FAISS_PATH="vectorestore/db_faiss"
def create_faiss_db(text_chunks,embeddings_model):
    db=FAISS.from_documents(text_chunks,embeddings_model)
    db.save_local(DB_FAISS_PATH)

create_faiss_db(text_chunks=text_chunks,embeddings_model=embeddings_model)

