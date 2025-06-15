import gradio as gr
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

genai_api_key = os.getenv("GOOGLE_API_KEY", "")
if not genai_api_key:
    print("Warning: GOOGLE_API_KEY not found. Please set it in your .env file or Hugging Face Space secrets.")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=genai_api_key, temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CV_FILE_PATH = "your_cv.pdf"

def setup_rag_pipeline(cv_path):
    print(f"Loading CV from {cv_path}...")
    try:
        loader = PyPDFLoader(cv_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No content found in the PDF. Is the file path correct and the PDF readable?")
        print(f"Loaded {len(documents)} pages from CV.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None, None

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating FAISS vector store from chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("FAISS vector store created.")

    print("RAG pipeline setup complete.")
    return vectorstore.as_retriever(search_kwargs={"k": 3}), vectorstore

retriever, vectorstore_instance = setup_rag_pipeline(CV_FILE_PATH)

if retriever is None:
    print("RAG pipeline setup failed. The chatbot might not work as expected.")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
else:
    qa_chain = None

def chat_with_cv(user_question, chat_history):
    if qa_chain is None:
        return "Chatbot configuration error: The CV could not be processed. Please contact the administrator."
    if not genai_api_key:
        return "Google API key missing. The chatbot cannot function. Please configure it."

    try:
        result = qa_chain.invoke({"question": user_question})
        ai_response = result["answer"]
        return ai_response
    except Exception as e:
        print(f"Error during chatbot interaction: {e}")
        if "Blocked by Google" in str(e):
             return "Sorry, I couldn't generate a response. The request was blocked by the Google model (possibly due to inappropriate or abusive content)."
        return "Sorry, an error occurred while generating the response. Please try again."

iface = gr.ChatInterface(
    fn=chat_with_cv,
    title="AI CV Assistant: Interrogate My Professional Journey",
    description="Ask questions about my CV (experiences, skills, education, projects). I remember our conversation!",
    examples=[
        "Tell me about your Machine Learning experience.",
        "What are your key NLP skills?",
        "Where did you study Data Science?",
        "Describe your fraud detection project.",
        "What is your experience at Amazon?",
        "What languages do you speak?"
    ],
    chatbot=gr.Chatbot(height=400),
    theme="soft",
)

if __name__ == "__main__":
    iface.launch()
