import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    chunk_size=1000,
    max_retries=3,
    model="embedding",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def process_pdf(pdf_file):
    """Process PDF and store in ChromaDB"""
    pdf_reader = PdfReader(pdf_file)
    
    # Extract text from PDF
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create and store embeddings in ChromaDB
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()
    return vectorstore

def get_qa_chain(vectorstore):
    """Create QA chain"""
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=PROMPT
    )
    return qa_chain, vectorstore.as_retriever()

# Streamlit UI
st.title("📚 PDF Question Answering System")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = False

# File upload section
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if not st.session_state.processed_files:
        with st.spinner("Processing PDF..."):
            try:
                vectorstore = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = True
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# Question answering section
if st.session_state.processed_files:
    st.header("2. Ask Questions")
    question = st.text_input("Enter your question about the PDF:")
    
    if question:
        if st.button("Get Answer"):
            try:
                with st.spinner("Thinking..."):
                    # Load vector store and create QA chain
                    qa_chain, retriever = get_qa_chain(st.session_state.vectorstore)
                    
                    # Get relevant documents
                    docs = retriever.get_relevant_documents(question)
                    
                    # Get answer
                    response = qa_chain(
                        {
                            "input_documents": docs,
                            "question": question
                        }
                    )
                    
                    # Display answer in a nice format
                    st.write("### Answer:")
                    st.write(response["output_text"])
                    
                    # Display source documents
                    st.write("### Sources:")
                    for i, doc in enumerate(docs):
                        st.write(f"Source {i+1}:")
                        st.write(doc.page_content)
                    
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

# Clear conversation button
if st.session_state.processed_files:
    if st.button("Clear PDF and Start Over"):
        st.session_state.processed_files = False
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
        st.experimental_rerun()

# Add some usage instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a PDF file using the file uploader
    2. Wait for the PDF to be processed
    3. Enter your question about the PDF content
    4. Click 'Get Answer' to receive a response
    5. Use 'Clear PDF and Start Over' to process a new PDF
    """)
    
    st.header("About")
    st.markdown("""
    This app uses:
    - Azure OpenAI for embeddings and answering
    - ChromaDB for vector storage
    - LangChain for the QA chain
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")