

import os
import re
import numpy as np
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS  # Fixed deprecated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Set your Google API key here
API_KEY = "AIzaSyB2A739KlrZav3cRZ_Jk367EPYcO_f4prQ"  # Replace with your actual API key

# Set page configuration
st.set_page_config(
    page_title="Sustainability Knowledge Assistant",
    page_icon="🌱",
    layout="wide"
)

# App title and description
st.title("🌱 Sustainability Knowledge Assistant")
st.markdown("""
    Ask questions about sustainability topics or perform calculations.
    This assistant uses Google's Gemini AI model to provide information about climate change, 
    recycling, sustainable energy, and other environmental topics.
""")

# Sidebar for mode selection and information
with st.sidebar:
    st.header("Mode Selection")
    query_mode = st.radio(
        "Select Mode:",
        ["Knowledge Base", "Direct Gemini Query", "Calculator"],
        index=0
    )
    
    st.header("About")
    st.markdown("""
    This application provides information about sustainability topics using:
    - A curated knowledge base on environmental topics
    - Google's Gemini Pro AI model
    - Basic calculation capabilities
    """)

# Function to initialize Gemini and LangChain components
@st.cache_resource
def initialize_components():
    # Set API key
    os.environ["GOOGLE_API_KEY"] = API_KEY
    genai.configure(api_key=API_KEY)
    
    # Initialize LLM
    llm = GoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0.5,
        max_output_tokens=512
    )
    
    # Sample sustainability data
    documents = [
        "Climate change refers to significant changes in global temperatures and weather patterns over time.",
        "Recycling helps reduce waste by converting used materials into new products.",
        "Carbon footprint is the total greenhouse gas emissions caused by an individual or organization.",
        "Sustainable energy includes solar, wind, and hydroelectric power.",
        "Reducing plastic usage is essential for minimizing ocean pollution.",
        "Biodiversity is crucial for maintaining ecological balance and ecosystem services.",
        "Deforestation contributes to climate change by reducing carbon sinks.",
        "Green buildings use sustainable materials and energy-efficient designs.",
        "Water conservation involves reducing water usage and preventing water pollution.",
        "Sustainable agriculture aims to meet society's food needs while preserving natural resources."
    ]
    
    # Data Ingestion and Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    
    # Create vector store
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embedding_model)
    
    # Create prompt template
    template = """
    You are a helpful assistant that provides information about sustainability and environmental topics.
    Based on the following context, please answer the question. If you don't know the answer from the context, 
    say that you don't have enough information rather than making something up.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retriever QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return llm, qa_chain

# Calculator function
def calculator(query):
    expression = re.search(r'calculate\s+([\d\+\-\*\/\(\)\.\s]+)', query.lower())
    
    if not expression and any(op in query for op in ['+', '-', '*', '/', '=']):
        numbers = re.findall(r'\d+', query)
        if len(numbers) == 2 and '+' in query:
            result = int(numbers[0]) + int(numbers[1])
            return f"The result of {numbers[0]} + {numbers[1]} is {result}"
    
    if not expression:
        return "Could not find a valid mathematical expression. Please use format: calculate 2 + 2"
    
    try:
        expression = expression.group(1).strip()
        if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
            return "Invalid characters in expression."
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"

# Direct Gemini query function
def direct_gemini_query(query):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    response = model.generate_content(query)
    return response.text

# Main query input
query = st.text_input("Enter your question or calculation:", placeholder="E.g., 'What is climate change?' or 'calculate 24 * 7'")

# Process query when submitted
if query:
    try:
        # Initialize components
        llm, qa_chain = initialize_components()
        
        # Show a spinner while processing
        with st.spinner("Processing your query..."):
            # Determine which mode to use
            use_calculator = query_mode == "Calculator" or (query_mode == "Knowledge Base" and 
                                                        (query.lower().startswith("calculate") or 
                                                         any(op in query for op in ['+', '-', '*', '/', '='])))
            
            if use_calculator:
                st.subheader("🧮 Calculator Result")
                result = calculator(query)
                st.success(result)
                
            elif query_mode == "Direct Gemini Query":
                st.subheader("🤖 Gemini Response")
                try:
                    result = direct_gemini_query(query)
                    st.write(result)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    
            else:  # Knowledge Base mode
                st.subheader("📚 Knowledge Base Response")
                try:
                    result = qa_chain.invoke({"query": query})
                    
                    # Display answer
                    st.markdown("### Answer:")
                    st.write(result['result'])
                    
                    # Display sources in an expander
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Document {i+1}:** {doc.page_content}")
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}. Please check the API key configuration.")

# Add some helpful examples
st.markdown("---")
st.subheader("Example queries you can try:")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Knowledge Base:**
    - What is climate change?
    - Why is recycling important?
    - What are sustainable energy sources?
    """)
with col2:
    st.markdown("""
    **Direct Gemini Queries:**
    - What are the latest sustainability trends?
    - How can I reduce my carbon footprint?
    - Explain the Paris Climate Agreement
    """)
with col3:
    st.markdown("""
    **Calculations:**
    - calculate 2025 - 1990
    - calculate 24 * 7
    - what is 145 + 376?
    """)

# Footer
st.markdown("---")
st.caption("Sustainability Knowledge Assistant powered by Google Gemini AI")