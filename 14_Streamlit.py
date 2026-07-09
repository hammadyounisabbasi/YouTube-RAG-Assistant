# YouTube RAG Assistant
# Streamlit + LangChain + Hugging Face + FAISS
# Imports
import os
import re
import time
from dotenv import load_dotenv
import streamlit as st
# YouTube Transcript API
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

# Hugging Face
from huggingface_hub import InferenceClient
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

# LangChain
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import (
    FAISS,
)
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import (
    StrOutputParser,
)

# Streamlit Configuration
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Environment Variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    st.error(
        "HUGGINGFACE_API_TOKEN not found in .env file."
    )
    st.stop()

# Required by LangChain HuggingFaceEndpoint
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
# Hugging Face Inference Client
# (Used only for translation)
hf_client = InferenceClient(
    api_key=HF_TOKEN
)

# Session State Initialization
DEFAULT_SESSION_STATE = {
    # Vector Database
    "vector_store": None,
    # RAG Chain
    "rag_chain": None,
    # Chat Messages
    "messages": [],
    # Current Video
    "video_loaded": False,
    "video_id": "",
    "video_title": "",
    # Transcript
    "transcript": "",
    "video_language": "en",
    "video_language_name": "English",
    # Translation Cache
    "translation_cache": {},
}
for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Supported Languages
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# App Header
st.title("🎥 YouTube RAG Assistant")
st.caption(
    "Ask questions about any YouTube video using Retrieval-Augmented Generation (RAG)."
)

# Helper Functions
def extract_video_id(url_or_id: str):
    """
    Supports:
    https://www.youtube.com/watch?v=xxxx
    https://youtu.be/xxxx
    https://www.youtube.com/embed/xxxx
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

# Load Transcript
def load_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    transcripts = list(transcript_list)
    if len(transcripts) == 0:
        raise Exception(
            "No transcript available for this video."
        )
    selected = None
    
    # Priority 1
    # Manual English
    for transcript in transcripts:
        if (
            transcript.language_code == "en"
            and not transcript.is_generated
        ):
            selected = transcript
            break

    # Priority 2
    # Auto English
    if selected is None:
        for transcript in transcripts:
            if transcript.language_code == "en":
                selected = transcript
                break

    # Priority 3
    # Manual Any Language
    if selected is None:
        for transcript in transcripts:
            if not transcript.is_generated:
                selected = transcript
                break

    # Priority 4
    # First Available
    if selected is None:
        selected = transcripts[0]
    fetched = api.fetch(
        video_id,
        languages=[selected.language_code]
    )
    transcript_text = " ".join(
        chunk.text
        for chunk in fetched
    )
    return {
        "text": transcript_text,
        "language_code": selected.language_code,
        "language_name": selected.language,
        "generated": selected.is_generated,
    }

# Cached Embedding Model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={
            "token": HF_TOKEN
        }
    )

# Build Vector Store
def build_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.create_documents(
        [transcript_text]
    )
    embeddings = load_embedding_model()
    vector_store = FAISS.from_documents(
        docs,
        embeddings,
    )
    return vector_store

# Cached Hugging Face LLM
@st.cache_resource
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.2,
        max_new_tokens=512,
    )
    model = ChatHuggingFace(
        llm=endpoint
    )
    return model

# Translate Retrieved Context Only
def translate_context(context, language):
    """
    Only translate retrieved chunks.
    English transcript:
        Returns original context.
    Other languages:
        Uses Hugging Face Llama 3.1 for translation.
    """
    if language == "en":
        return context
    cache = st.session_state.setdefault(
    "translation_cache",
    {}
)
    if context in cache:
        return cache[context]
    prompt = f"""
You are a professional translator.
Translate the following text into fluent English.
Rules:
1. Return ONLY the translated English text.
2. Do NOT summarize.
3. Do NOT explain.
4. Preserve technical terms.
5. Preserve formatting.
6. Preserve names.
7. Preserve code exactly.
Text:
{context}
"""
    try:
        response = hf_client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=1024,
        )
        translated = response.choices[0].message.content
        cache[context] = translated
        return translated
    except Exception:
        return context

# Prompt Template
prompt = PromptTemplate(
    template="""
You are an intelligent AI assistant.
The transcript context below has already been translated into English if necessary.
Answer ONLY using the provided context.
If the answer is not present in the context, reply exactly:
I don't know.
Rules:
- Do not make up information.
- Keep answers concise.
- Answer in English.
- If the context is insufficient, say:
I don't know.
Context:
{context}
Question:
{question}
Answer:
""",
    input_variables=[
        "context",
        "question",
    ],
)

# Format Retrieved Documents
def format_docs(docs, language):
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    context = translate_context(
        context,
        language,
    )
    return context

# Build RAG Chain
def build_rag_chain(
    vector_store,
    language,
):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        },
    )
    parallel_chain = RunnableParallel(
        {
            "context":
                retriever
                | RunnableLambda(
                    lambda docs: format_docs(
                        docs,
                        language,
                    )
                ),
            "question":
                RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    rag_chain = (
        parallel_chain
        | prompt
        | load_llm()
        | parser
    )
    return rag_chain

# Utility Function
def reset_knowledge_base():
    st.session_state.vector_store = None
    st.session_state.rag_chain = None
    st.session_state.video_loaded = False
    st.session_state.video_id = ""
    st.session_state.video_title = ""
    st.session_state.transcript = ""
    st.session_state.video_language = "en"
    
    if "translation_cache" not in st.session_state:
       st.session_state.translation_cache = {}
    else:
       st.session_state.translation_cache.clear()

    st.session_state.video_language_name = "English"
    st.session_state.translation_cache = {}
    st.session_state.messages = []

# Build Knowledge Base
def create_knowledge_base(video_input):
    video_id = extract_video_id(video_input)
    transcript_data = load_transcript(video_id)
    transcript = transcript_data["text"]
    language_code = transcript_data["language_code"]
    language_name = transcript_data["language_name"]
    vector_store = build_vector_store(
        transcript
    )
    rag_chain = build_rag_chain(
        vector_store,
        language_code,
    )
    st.session_state.vector_store = vector_store
    st.session_state.rag_chain = rag_chain
    st.session_state.video_loaded = True
    st.session_state.video_id = video_id
    st.session_state.transcript = transcript
    st.session_state.video_language = language_code
    st.session_state.video_language_name = language_name
    st.session_state.translation_cache = {}
    st.session_state.messages = []
    return True

# Sidebar
with st.sidebar:
    st.title("🎥 YouTube RAG")
    st.markdown("---")
    youtube_input = st.text_input(
        "Paste YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    build_btn = st.button(
        "🚀 Build Knowledge Base",
        use_container_width=True,
    )
    st.markdown("---")
    if st.session_state.video_loaded:
        st.success("Knowledge Base Ready")
        st.write(
            f"**Video ID:** {st.session_state.video_id}"
        )
        st.write(
            f"**Transcript Language:** {st.session_state.video_language_name}"
        )
    else:
        st.info("No video loaded.")
    st.markdown("---")
    st.info(
        """
### Features
✅ YouTube URL
✅ Video ID
✅ Automatic Transcript Detection
✅ Multilingual Retrieval
✅ English Answers
✅ Hugging Face Llama 3.1
✅ FAISS Vector Database
✅ LangChain RAG
"""
    )

# Build Knowledge Base
if build_btn:
    if youtube_input.strip() == "":
        st.warning(
            "Please enter a YouTube URL or Video ID."
        )
    else:
        try:
            with st.spinner(
                "Downloading transcript..."
            ):
                create_knowledge_base(
                    youtube_input
                )
            st.success(
                "Knowledge Base Created Successfully!"
            )
            st.balloons()
            st.rerun()
        except TranscriptsDisabled:
            st.error(
                "Transcripts are disabled for this video."
            )
        except NoTranscriptFound:
            st.error(
                "No transcript found for this video."
            )
        except Exception as e:
            st.error(str(e))

# Video Information
if st.session_state.video_loaded:
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(
            f"https://img.youtube.com/vi/{st.session_state.video_id}/hqdefault.jpg",
            use_container_width=True,
        )
    with col2:
        st.subheader("Current Video")
        st.write(
            f"**Video ID:** {st.session_state.video_id}"
        )
        st.write(
            f"**Transcript Language:** {st.session_state.video_language_name}"
        )
        st.write(
            f"**Messages:** {len(st.session_state.messages)}"
        )
        st.write(
            f"**Translation Cache:** {len(st.session_state.translation_cache)} items"
        )
        with st.expander("Preview Transcript"):
            st.text_area(
                "Transcript",
                value=st.session_state.transcript[:5000],
                height=250,
                disabled=True,
            )
st.divider()

# Chat Interface
st.subheader("💬 Chat with your YouTube Video")
if not st.session_state.video_loaded:
    st.info(
        "👈 Enter a YouTube URL in the sidebar and click **Build Knowledge Base**."
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
                "role": "user",
                "content": question,
            }
        )
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    answer = st.session_state.rag_chain.invoke(
                        question
                    )
                    elapsed = time.time() - start_time
                    placeholder.markdown(answer)
                    st.caption(
                        f"⏱ Response generated in {elapsed:.2f} sec"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )
                except Exception as e:
                    placeholder.error(str(e))

# Utility Buttons
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button(
        "🗑 Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button(
        "♻ Reset Knowledge Base",
        use_container_width=True,
    ):
        reset_knowledge_base()
        st.rerun()

# Download Chat History
if st.session_state.messages:
    chat_text = "# YouTube RAG Chat History\n\n"
    for msg in st.session_state.messages:
        role = msg["role"].capitalize()
        chat_text += f"## {role}\n"
        chat_text += msg["content"]
        chat_text += "\n\n"
    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="youtube_rag_chat.md",
        mime="text/markdown",
        use_container_width=True,
    )

# Sidebar Session Information
with st.sidebar:
    if st.session_state.video_loaded:
        st.markdown("---")
        st.subheader("📊 Session Information")
        st.write(
            f"**Transcript Language:** {st.session_state.video_language_name}"
        )
        st.write(
            f"**Messages:** {len(st.session_state.messages)}"
        )
        st.write(
            f"**Translation Cache:** {len(st.session_state.translation_cache)}"
        )
        st.write("**Embedding Model**")
        st.caption("intfloat/multilingual-e5-small")
        st.write("**LLM**")
        st.caption("Meta-Llama-3.1-8B-Instruct")
        st.write("**Vector Store**")
        st.caption("FAISS")
        st.write("**Framework**")
        st.caption("LangChain")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align:center">
Retrieval-Augmented Generation (RAG) application built with:

**LangChain • Hugging Face • Meta Llama 3.1 • FAISS • Streamlit • YouTube Transcript API**

### Features

✅ Supports YouTube URL or Video ID

✅ Automatic transcript detection

✅ Multilingual transcript support

✅ Retrieves transcript chunks in the original language

✅ Translates only the retrieved context into English

✅ Answers generated using Meta Llama 3.1

✅ FAISS semantic search
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "Developed by Hammad Younis Abbasi"
)
