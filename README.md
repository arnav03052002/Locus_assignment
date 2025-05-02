# Locus Survey AI Assistant

An interactive AI-powered assistant that answers questions about the Locus Platform Survey Implementation Guide using Retrieval-Augmented Generation (RAG).

Built with:
- FAISS for vector similarity search
- Groq LLM API for language generation
- BAAI/bge-base-en-v1.5 for sentence embeddings
- Streamlit for the user interface

## Features

- Document-based question answering using the official Locus Survey PDF
- Natural language chat interface built with Streamlit and Groq
- Retrieved document chunks shown with every answer
- Chat sessions saved with per-query context
- Export chat history as `.txt` or `.json` for documentation or auditing

## Tech Stack

- Python 3.12+
- Streamlit
- FAISS
- Hugging Face Transformers
- Groq API
- PyMuPDF

## Folder Structure


├── app.py                      # Streamlit chatbot interface <br>
├── model.py                    # PDF chunking + FAISS index builder<br>
├── vector.index                # Precomputed FAISS index<br>
├── chunks.json                 # Document chunks for retrieval<br>
├── chat_sessions/              # Saved session transcripts<br>
├── requirements.txt            # Dependencies<br>
└── Locus Platform Survey Implementation Guide 1.pdf<br>

## How It Works

1. `model.py`:
    - Loads the PDF
    - Splits it into text chunks
    - Embeds the chunks using `BAAI/bge-base-en-v1.5`
    - Stores the embeddings in a FAISS index (`vector.index`)

2. `app.py`:
    - Embeds user questions
    - Finds relevant chunks via FAISS similarity search
    - Builds a prompt with top chunks and sends it to the Groq LLM
    - Returns a human-friendly answer
    - Shows retrieved context and stores it with the question

## Setup Instructions

1. Clone this repository

```
git clone https://github.com/your-username/locus-survey-ai-assistant.git
cd your_folder_name
```

2. Create and activate a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

3. Install all dependencies

```
pip install -r requirements.txt
```

4. Add your Groq API key to a `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the application

```
streamlit run app.py
```


