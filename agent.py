
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_session_id
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langsmith import Client
# Initialize the document loader
file_paths = ["repealedfileopen.pdf"]
docs = []
for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())  #

# Configure Gemini Embeddings
genai.configure(api_key="AIzaSyAJplRNyOItvjSUDqoE8yAtyFeb7wo9yZE")

class GeminiEmbeddings:
    def __init__(self, model_name="models/text-embedding-004", api_key="AIzaSyAJplRNyOItvjSUDqoE8yAtyFeb7wo9yZE"):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed_query(self, text):
        result = genai.embed_content(model=self.model_name, content=text)
        return result['embedding']

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

embeddings = GeminiEmbeddings()

# Split and vectorize the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="./chroma_db")

# Export retriever object for reuse
retriever = vectorstore.as_retriever()

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key="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")

# Create the prompt
system_prompt = (
    '''"You are an expert legal assistant for question-answering tasks based on the Indian Penal Code (IPC). Your primary role is to provide accurate, clear, and concise information about IPC sections, definitions, punishments, legal procedures, and related jurisprudence.

Use the following guidelines:

Accuracy: Rely strictly on the retrieved context from the IPC document. Do not speculate or provide interpretations beyond the text.

Clarity: Break down complex legal terms into simple language for better understanding.

Conciseness: Prioritize direct answers but include relevant details (e.g., section numbers, exceptions, precedents) where necessary.

Scope: Cover IPC provisions—including crimes, penalties, defenses, and amendments—but avoid advising on specific cases or personal legal advice.

Transparency: If a query is outside the IPC (e.g., civil laws, state-specific rules) or unclear, state that you cannot assist or need more context.

Example Topics:

Definitions (e.g., theft, murder, assault).

Punishments (imprisonment, fines, death penalty).

Landmark judgments linked to IPC sections.

Differences between similar offenses (e.g., culpable homicide vs. murder).

Recent amendments or explanations of controversial sections.* '''
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Set up the retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = rag_chain.invoke(
        {"input": user_input},)

    return response['answer']