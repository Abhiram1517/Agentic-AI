
#Step 1: Setup LLM Mistral with HuggingFace
import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace


# Load environment variables
load_dotenv()
# Step 1: Setup LLM Mistral with HuggingFace
HF_TOKEN=os.getenv("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.2"
# HUGGINGFACE_REPO_ID = "google/flan-t5-large"
HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"



# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         huggingfacehub_api_token=HF_TOKEN,
#         max_new_tokens=512
#     )
#     return llm
# def load_llm(repo_id, token):
#     llm = ChatHuggingFace(
#         repo_id=repo_id,
#         huggingfacehub_api_token=token,
#         temperature=0.5,
#         max_new_tokens=512
#     )
#     return llm

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def load_llm(repo_id, token):

    # Step 1: Base LLM (endpoint)
    base_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        temperature=0.5,
        max_new_tokens=512
    )

    # Step 2: Wrap in chat model
    chat_llm = ChatHuggingFace(llm=base_llm)

    return chat_llm



#Step 2: Connect llm WITH FAISS and create chain to query the memory and get response from LLM
CUSTOM_PROMPT_TEMPLATE="""
Use the following pieces of context to answer the question at the end. If you don't know the answer, say you don't know.
don't make up an answer. Always use all the relevant information you have access to. Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt_template(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

#Load Database
DB_FAISS_PATH="vectorestore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

# Create RetrievalQA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID,HF_TOKEN),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True, 
    chain_type_kwargs={"prompt": set_custom_prompt_template(CUSTOM_PROMPT_TEMPLATE)}
    )

# Now invoke with the user query
user_query=input("Enter your question: ")
response = qa_chain.invoke({"query": user_query})
print("ANSWER: ",response['result'])
print("SOURCE DOCUMENTS: ",response['source_documents'])