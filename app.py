from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS   # ✅ replaced Chroma with FAISS
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
/* Your full CSS stays same as before */
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
                
                # ✅ Create FAISS vector store
                vectorstore = FAISS.from_documents(splits, embeddings)
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
                
                # ✅ Sync: Add user message to both UI & LangChain memory
                st.session_state.chat_history.append(("user", user_input, timestamp))
                session_history = get_session_history(session_id)
                session_history.add_user_message(user_input)
                
                # Process the query
                with st.spinner("🤔 Analyzing your question..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        assistant_msg = response['answer']
                        
                        # ✅ Sync: Add assistant response to both UI & LangChain memory
                        st.session_state.chat_history.append(("assistant", assistant_msg, timestamp))
                        session_history.add_ai_message(assistant_msg)
                        
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
