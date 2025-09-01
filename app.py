from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
import time
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IntelliDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        min-height: 100vh;
    }
    
    /* Custom header */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar styling - matching title bar */
    .css-1d391kg {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border-right: 1px solid #475569;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar content styling */
    .css-1d391kg .stMarkdown, 
    .css-1d391kg label,
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stTextInput label,
    .css-1d391kg .stFileUploader label,
    .css-1d391kg p,
    .css-1d391kg div {
        color: #ffffff !important;
    }
    
    /* Sidebar headers and text */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
    }
    
    /* Section titles in sidebar */
    section[data-testid="stSidebar"] .section-title {
        color: #ffffff !important;
        border-bottom: 2px solid #475569 !important;
    }
    
    /* File uploader in sidebar */
    section[data-testid="stSidebar"] .stFileUploader {
        background: rgba(248, 250, 252, 0.1);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #475569;
    }
    
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Section containers */
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .section-title {
        color: #1e293b !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin: 0 0 1rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    
    /* Main content text styling */
    .section-container p {
        color: #1e293b !important;
        line-height: 1.6 !important;
        margin: 1rem 0 !important;
    }
    
    .section-container h4 {
        color: #1e293b !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 0.5rem 0 !important;
    }
    
    .section-container ul {
        color: #1e293b !important;
        padding-left: 1.5rem !important;
    }
    
    .section-container li {
        color: #1e293b !important;
        margin: 0.5rem 0 !important;
        line-height: 1.5 !important;
    }
    
    .section-container strong {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background: #f8fafc !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
        color: #1e293b !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Sidebar input fields */
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(248, 250, 252, 0.1) !important;
        border: 2px solid #475569 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f8fafc !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .uploadedFile:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
    }
    
    /* Sidebar file uploader specific */
    section[data-testid="stSidebar"] .stFileUploader > div {
        background: rgba(248, 250, 252, 0.1) !important;
        border: 2px dashed #475569 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: rgba(248, 250, 252, 0.1) !important;
        border: 2px dashed #475569 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #60a5fa !important;
        background: rgba(96, 165, 250, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzone"] span {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stFileUploader small {
        color: #94a3b8 !important;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
    }
    
    .assistant-message {
        background: #f8fafc;
        color: #1e293b;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Status indicators */
    .status-success {
        background: #dcfce7;
        color: #166534;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 80px;
    }
    
    .loading-dots div {
        position: absolute;
        top: 33px;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        background: #3b82f6;
        animation-timing-function: cubic-bezier(0, 1, 1, 0);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">IntelliDoc AI</h1>
    <p class="header-subtitle">Advanced Document Intelligence & Conversational AI Platform</p>
</div>
""", unsafe_allow_html=True)

# Initialize session states
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = 0
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="section-title">🔧 Configuration</div>', unsafe_allow_html=True)
    
    # API Key input
    st.markdown("**Groq API Key**")
    groq_api = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        placeholder="gsk_...",
        help="Your Groq API key for accessing the language model"
    )
    
    # Session management
    st.markdown("**Session Management**")
    session_id = st.text_input(
        "Session ID:",
        value="default_session",
        help="Unique identifier for your chat session"
    )
    
    # Document upload
    st.markdown("**📄 Document Upload**")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents:",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to analyze"
    )
    
    # Display metrics
    if st.session_state.documents_processed > 0:
        st.markdown("**📊 Session Statistics**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.documents_processed}</div>
                <div class="metric-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.total_chunks}</div>
                <div class="metric-label">Text Chunks</div>
            </div>
            """, unsafe_allow_html=True)

# Main content area
if not groq_api:
    st.markdown("""
    <div class="status-warning">
        <strong>⚠️ Setup Required</strong><br>
        Please enter your Groq API key in the sidebar to begin using IntelliDoc AI.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-container">
        <div class="section-title">🚀 Getting Started</div>
        <p>IntelliDoc AI is a powerful document intelligence platform that allows you to:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    - **Upload PDF documents** and extract meaningful insights
    - **Ask questions** about your document content using natural language
    - **Maintain conversation context** across multiple queries
    - **Get accurate answers** backed by your document sources
    """)
    
    st.markdown("To get started, simply add your Groq API key in the sidebar and upload your first PDF document.")
else:
    # Initialize the language model
    try:
        llm = ChatGroq(groq_api_key=groq_api, model="llama-3.3-70b-versatile")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.markdown("""
        <div class="status-success">
            <strong>✅ System Ready</strong><br>
            IntelliDoc AI is configured and ready to process your documents.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div class="status-error">
            <strong>❌ Configuration Error</strong><br>
            {str(e)}
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Document processing
    if uploaded_files:
        with st.spinner("🔄 Processing documents..."):
            try:
                documents = []
                processed_files = []
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    temp_path = f"./{uploaded_file.name}"
                    with open(temp_path, "wb") as file:
                        file.write(uploaded_file.getvalue())
                    
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    processed_files.append(uploaded_file.name)
                    
                    # Clean up temporary file
                    os.remove(temp_path)

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000, 
                    chunk_overlap=500
                )
                splits = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever()
                
                # Update session state
                st.session_state.documents_processed = len(processed_files)
                st.session_state.total_chunks = len(splits)
                
                # Set up the RAG chain
                contextual_q_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question that can be understood "
                    "without the chat history. Do not answer the question, "
                    "just reformulate it if needed and otherwise return it as is."
                )

                contextual_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextual_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextual_q_prompt
                )

                # QA System prompt
                system_prompt = (
                    "You are a professional AI assistant specialized in document analysis. "
                    "Use the following pieces of retrieved context to provide accurate, "
                    "well-structured answers. If you don't know something, clearly state that. "
                    "Always maintain a professional tone and cite relevant sections when possible.\n\n"
                    "Context: {context}"
                )

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # History management
                def get_session_history(session: str) -> BaseChatMessageHistory:
                    if session not in st.session_state.store:
                        st.session_state.store[session] = ChatMessageHistory()
                    return st.session_state.store[session]

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain, 
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_message_key="answer"
                )

                st.markdown(f"""
                <div class="status-success">
                    <strong>📄 Documents Processed Successfully</strong><br>
                    Processed {len(processed_files)} document(s) into {len(splits)} searchable chunks.<br>
                    <em>Files: {', '.join(processed_files)}</em>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                    <strong>❌ Document Processing Error</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.stop()

        # Chat Interface
        st.markdown("""
        <div class="section-container">
            <div class="section-title">💬 Document Q&A Interface</div>
        </div>
        """, unsafe_allow_html=True)

        # Display chat history
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if st.session_state.chat_history:
                for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>You</strong> <small>({timestamp})</small><br>
                            {message}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="assistant-message">
                            <strong>IntelliDoc AI</strong> <small>({timestamp})</small><br>
                            {message}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; color: #64748b; font-style: italic; padding: 2rem;">
                    Start a conversation by asking a question about your uploaded documents.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Query input
        with st.container():
            user_input = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is the main topic discussed in the document?",
                key="user_query"
            )
            
            if user_input and st.button("Send Query", type="primary"):
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add user message to chat history
                st.session_state.chat_history.append(("user", user_input, timestamp))
                
                # Process the query
                with st.spinner("🤔 Analyzing your question..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append(
                            ("assistant", response['answer'], timestamp)
                        )
                        
                        # Rerun to update the display
                        st.rerun()
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="status-error">
                            <strong>❌ Query Processing Error</strong><br>
                            {str(e)}
                        </div>
                        """, unsafe_allow_html=True)

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                if session_id in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                st.rerun()

    else:
        st.markdown("""
        <div class="section-container">
            <div class="section-title">📤 Upload Documents to Begin</div>
            <p>Upload your PDF documents using the file uploader in the sidebar to start analyzing and asking questions about your content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ✨ Key Features:")
        st.markdown("""
        - **Multi-document support:** Upload and analyze multiple PDFs simultaneously
        - **Contextual conversations:** Maintain conversation history across queries  
        - **Intelligent retrieval:** Advanced semantic search finds relevant information
        - **Professional responses:** Get well-structured, accurate answers
        """)

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 3rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
    <strong>IntelliDoc AI</strong> • Advanced Document Intelligence Platform<br>
    Powered by Groq & LangChain • Built with Streamlit
</div>
""", unsafe_allow_html=True)