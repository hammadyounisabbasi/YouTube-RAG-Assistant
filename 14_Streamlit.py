# Streamlit YouTube RAG App

import os
import re
from dotenv import load_dotenv

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# Page Config

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
)

# Load Environment Variables

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    st.error("HUGGINGFACE_API_TOKEN not found in .env")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Session State

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

# Helper Functions

def extract_video_id(url_or_id: str):
    """
    Accepts either:
    https://youtube.com/watch?v=xxxx
    https://youtu.be/xxxx
    xxxx
    """

    url_or_id = url_or_id.strip()

    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"embed/([^?&]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return url_or_id


def load_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    languages = []

    for item in transcript_list:
        languages.append(item.language_code)

    if not languages:
        raise Exception("No transcript found.")

    transcript = api.fetch(
        video_id,
        languages=[languages[0]]
    )

    text = " ".join(
        chunk.text
        for chunk in transcript
    )

    return text


def build_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )

    return vector_store

# Sidebar

with st.sidebar:

    st.title("🎥 YouTube RAG")

    st.markdown("---")

    youtube_input = st.text_input(
        "Paste YouTube URL or Video ID"
    )

    build_btn = st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True
    )

    st.markdown("---")

    st.info(
        """
        **Supported**

        ✅ YouTube URL

        ✅ Video ID

        ✅ Automatic transcript detection

        ✅ Hugging Face Llama 3.1

        ✅ FAISS Vector Database
        """
    )

# Main Page

st.title("🎥 YouTube RAG Assistant")

st.caption("Ask questions from any YouTube video.")

# Build Vector Database

if build_btn:

    if youtube_input.strip() == "":
        st.warning("Please enter a YouTube URL or Video ID.")

    else:

        video_id = extract_video_id(youtube_input)

        try:

            with st.spinner("Fetching Transcript..."):

                transcript = load_transcript(video_id)

            with st.spinner("Creating Embeddings..."):

                vector_store = build_vector_store(transcript)

            st.session_state.vector_store = vector_store

            st.session_state.video_loaded = True

            st.success("Knowledge Base Created Successfully!")

        except TranscriptsDisabled:

            st.error("Transcripts are disabled for this video.")

        except NoTranscriptFound:

            st.error("Transcript not available.")

        except Exception as e:

            st.error(str(e))

# Build LLM + RAG Chain

if (
    st.session_state.video_loaded
    and st.session_state.rag_chain is None
):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN,
    )

    model = ChatHuggingFace(llm=llm)

    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k":4}
    )

    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.

Answer ONLY from the transcript below.

If the transcript does not contain the answer,
reply exactly:

I don't know.

Transcript:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context","question"]
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    parser = StrOutputParser()

    st.session_state.rag_chain = (
        parallel_chain
        | prompt
        | model
        | parser
    )

# Chat Interface

st.divider()

st.subheader("💬 Chat with your YouTube Video")

if not st.session_state.video_loaded:

    st.info(
        "Enter a YouTube URL in the sidebar and click **Build Knowledge Base**."
    )

else:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask anything about this video..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = st.session_state.rag_chain.invoke(question)

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role":"assistant",
                            "content":answer
                        }
                    )

                except Exception as e:

                    st.error(str(e))

# Footer

st.divider()

col1,col2 = st.columns(2)

with col1:

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

with col2:

    if st.button("♻ Reset Knowledge Base"):

        st.session_state.vector_store = None
        st.session_state.rag_chain = None
        st.session_state.video_loaded = False
        st.session_state.messages = []

        st.rerun()

st.markdown(
    """
---
### 🚀 Tech Stack

- LangChain
- Hugging Face Llama 3.1
- Hugging Face Embeddings
- FAISS
- YouTube Transcript API
- Streamlit
"""
)