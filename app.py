import streamlit as st
st.set_page_config(page_title="Locus Survey AI Assistant")

import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
import io
import time
import faiss

# === Load environment and Groq API ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# === Ensure chat_sessions folder exists ===
if not os.path.exists("chat_sessions"):
    os.makedirs("chat_sessions")

# === Load FAISS index and chunks ===
index = faiss.read_index("vector.index")
with open("chunks.json", "r") as f:
    chunks = json.load(f)

# === Load embedding model ===
@st.cache_resource
def load_bge():
    model_name = "BAAI/bge-base-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu")
    return tokenizer, model

tokenizer, model = load_bge()

# === Embedding function ===
def embed_query(text):
    text = "Represent this sentence for retrieval: " + text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# === Session state init ===
def load_all_sessions():
    sessions = {}
    for fname in sorted(os.listdir("chat_sessions")):
        if fname.endswith(".json"):
            path = os.path.join("chat_sessions", fname)
            with open(path, "r") as f:
                sessions[fname.replace(".json", "")] = json.load(f)
    return sessions

def save_session(name, messages):
    path = os.path.join("chat_sessions", f"{name}.json")
    with open(path, "w") as f:
        json.dump(messages, f)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_all_sessions()
if "active_session" not in st.session_state:
    st.session_state.active_session = "Session 1"
if st.session_state.active_session not in st.session_state.chat_sessions:
    st.session_state.chat_sessions[st.session_state.active_session] = []

# === Sidebar UI ===
st.sidebar.title("Projects")
session_names = list(st.session_state.chat_sessions.keys())
selected = st.sidebar.radio("Select session:", session_names)

if selected != st.session_state.active_session:
    st.session_state.active_session = selected

if st.sidebar.button("New Chat"):
    i = 1
    while f"Session {i}" in st.session_state.chat_sessions:
        i += 1
    new_name = f"Session {i}"
    st.session_state.chat_sessions[new_name] = []
    st.session_state.active_session = new_name

st.sidebar.markdown("---")
st.sidebar.markdown("### Export Chat")

# === Export plain text chat
if st.sidebar.button("Export Current Session (.txt)"):
    chat_log = "\n\n".join(
        [f"User: {entry['query']}\nAssistant: {entry['answer']}" for entry in st.session_state.chat_sessions[st.session_state.active_session] if isinstance(entry, dict)]
    )
    filename = f"{st.session_state.active_session.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    buffer = io.BytesIO(chat_log.encode("ascii", errors="ignore"))
    buffer.seek(0)
    st.sidebar.download_button("Download chat", buffer, file_name=filename)

# ===  Export full session JSON
if st.sidebar.button("Download Full Session JSON"):
    json_str = json.dumps(st.session_state.chat_sessions[st.session_state.active_session], indent=2)
    buffer = io.BytesIO(json_str.encode("utf-8"))
    buffer.seek(0)
    st.sidebar.download_button(
        "Download JSON", 
        buffer, 
        file_name=f"{st.session_state.active_session.replace(' ', '_')}_session.json",
        mime="application/json"
    )

# === Main UI ===
st.title("Locus Survey AI Assistant")
st.caption(f"Current session: {st.session_state.active_session}")

session_chat = st.session_state.chat_sessions[st.session_state.active_session]

# === Show chat history with per-question chunks
for entry in session_chat:
    if isinstance(entry, dict):
        query = entry["query"]
        answer = entry["answer"]
        chunks_used = entry.get("chunks", [])
    else:
        query, answer = entry
        chunks_used = []

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(answer)
        if chunks_used:
            with st.expander("Show retrieved context"):
                for i, chunk in enumerate(chunks_used):
                    st.markdown(f"Chunk {i+1}:\n```\n{chunk.strip()}\n```")

# === Chat Input ===
query = st.chat_input("Ask something about the guide...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    try:
        with st.spinner("Thinking..."):
            query_vec = embed_query(query).reshape(1, -1)
            assert query_vec.shape[1] == index.d, f"Query dim {query_vec.shape[1]} != index dim {index.d}"

            # Retrieve top 20, filter top 15 by quality
            D, I = index.search(query_vec, k=20)
            scored_chunks = [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]
            filtered = [(text, score) for text, score in scored_chunks if len(text.strip()) > 100]
            top_chunks = [text for text, _ in sorted(filtered, key=lambda x: x[1])[:15]]

            if not top_chunks:
                st.warning("⚠️ No relevant content found in the document.")
                raise ValueError("Insufficient context")

            context = "\n\n".join(top_chunks)

            #  Friendly, natural assistant prompt
            prompt = f"""You're a helpful assistant answering questions based on the provided document content.

You are a helpful assistant answering questions based on the provided document content.

Use the retrieved context below to answer the user's question clearly and naturally. Structure your answer in steps, bullets, or concise paragraphs — whichever suits the question.

Do not make up information. Only use what is present in the context. If the answer is partially available, say so and answer as best you can.

If the user's message is emotional or expressive (e.g., "thanks", "ok", "got it", "cool", "sorry"), reply briefly — just like a normal human would. Be casual, friendly, and don't over-explain.

Context:
{context}

Question: {query}
Answer:"""

            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()

        with st.chat_message("assistant"):
            animated = ""
            message_placeholder = st.empty()
            for word in answer.split():
                animated += word + " "
                message_placeholder.markdown(animated)
                time.sleep(0.02)

        # Save entry
        session_chat.append({
            "query": query,
            "answer": answer,
            "chunks": top_chunks
        })
        save_session(st.session_state.active_session, session_chat)

    except Exception as e:
        st.error(f" Error: {e}")
