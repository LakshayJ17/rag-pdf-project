import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import uuid

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")

st.title("AskMyPDF - Chat with your PDF")

st.warning("You can ask up to 3 questions for free. Please enter your own OpenAI API key to continue chatting after that.")

# Session start
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# Indexing / Ingestion phase
uploaded_file = st.file_uploader("Upload your PDF file ", type="pdf")

if uploaded_file:
    # Reset state if a new file is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.messages = []
        st.session_state.pdf_ready = False
        st.session_state.vector_db = None
        st.session_state.last_uploaded_filename = uploaded_file.name

    st.success("PDF uploaded successfully!")

    prepare_button = st.button("Prepare this PDF")

    if prepare_button and not st.session_state.pdf_ready:
        file_name = f"{uuid.uuid4().hex}.pdf"
        pdf_path = Path(file_name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Preparing the file for chat..."):
            loader = PyPDFLoader(file_path=pdf_path)
            docs = loader.load()

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents=docs)

            # VE & storing in vector DB - Qdrant DB
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_key
            )

            vector_db = QdrantVectorStore.from_documents(
                documents=split_docs,
                url="https://a3619777-b68b-4f91-95ed-fbf0c0c28815.us-west-1-0.aws.cloud.qdrant.io",
                api_key=qdrant_api_key,
                collection_name=file_name,
                embedding=embedding_model
            )
            st.session_state.vector_db = vector_db
            st.session_state.pdf_ready = True
            st.success("PDF read successfully! Ready for chat.")        
else:
    st.info("Please upload a PDF to begin")

    
# Show inputbox and chats only if PDF is ready
# Retrieval phase
if st.session_state.get("pdf_ready", False):
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**🧑 You:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**🤖 Assistant:** {message['content']}")
    st.divider()

    # Count user messages
    user_msg_count = sum(1 for m in st.session_state.messages if m["role"] == "user")

    if user_msg_count >= 3 and not st.session_state.user_api_key:
        st.warning("You have used your 3 free messages. Please enter your own OpenAI API key to continue.")
        user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if user_api_key:
            st.session_state.user_api_key = user_api_key
            st.success("API key saved! You can continue chatting.")
            st.rerun() 
        st.stop()

    query = st.text_input("Ask your query :")
    if query:
        st.session_state.messages.append({"role" : "user", "content" : query})

        # user api key / default key
        api_key_to_use = st.session_state.user_api_key if st.session_state.user_api_key else api_key
        client = OpenAI(api_key=api_key_to_use)

        # Get related chunks from DB
        search_results = st.session_state.vector_db.similarity_search(query=query)

        if not search_results:
            st.warning("Oops ! No relevant data found...")
        else:
            context = "\n\n\n".join([
                f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
                for result in search_results
            ])

            SYSTEM_PROMPT = f"""
                You are a helpful AI Assistant who answers user query based on the available context
                retrieved from a PDF file along with page_contents and page number.

                You should only ans the user based on the following context and navigate the user
                to open the right page number to know more.
                If the answer is not in the context, politely say you don't know and suggest the user check the referenced pages for more details.

                Context:
                {context}
            """
            messages = [{"role" : "system", "content" : SYSTEM_PROMPT}]

            for message in st.session_state.messages[1:]:
                messages.append(message)

            chat_completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )

            assistant_reply = chat_completion.choices[0].message.content
            st.session_state.messages.append({"role" : "assistant", "content": assistant_reply})
            st.markdown("**🤖 Answer:**")
            st.write(assistant_reply)




