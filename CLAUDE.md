# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Server (requires GOOGLE_API_KEY env var for embedder initialization)
GOOGLE_API_KEY=<key> python manage.py runserver

# ASGI server with WebSocket support
GOOGLE_API_KEY=<key> daphne -b 0.0.0.0 -p 8000 Agent_Project.asgi:application

# Celery worker (requires Redis running on localhost:6379)
celery -A Agent_Project worker -l info

# Migrations
python manage.py makemigrations Agent_app
python manage.py migrate

# Create admin user
python manage.py createsuperuser
```

## Architecture

**Django 5.2 + DRF REST API** with two main directories:
- `Agent_Project/` — Django project config (settings, ASGI, Celery, URL root)
- `Agent_app/` — All application code (single app)

### Core Data Flow

There are two independent features sharing the same app:

**1. AI Agent Platform** (the main product):
- Admin creates an `Agent` with voice/presenter config → uploads PDF/CSV files
- `FileUploadView` saves files, queues `process_pdf_task` via Celery
- Celery task uses `PdfHandler`/`CSVHandler` to extract text → `Embedder` builds a per-agent FAISS index (`{agent_id}_faiss_index/`)
- Users query agents via WebSocket (`DIDProxyConsumer` at `ws/stream-avatar/`) or REST (`VoiceQueryAPIView`)
- Queries: audio → Whisper STT → FAISS similarity search → Gemini LLM answer → D-ID avatar video response

**2. Interview Assistant** (standalone feature):
- User uploads resume PDF → `PdfHandler` extracts text → Gemini generates 5 questions
- One-at-a-time Q&A: user answers (text or audio via Gemini STT) → Gemini analyzes + rates each answer → gTTS converts feedback to audio
- After all 5 answers, Gemini generates a summary with strengths/weaknesses/recommendations

### Key Modules

| Module | Role |
|---|---|
| `embedder.py` | FAISS vector store with Google Generative AI embeddings + spaCy Italian NLP preprocessing. Embedder is lazy-loaded (`get_embedder()`) to avoid blocking server startup. |
| `pdf_handler.py` | PyMuPDF-based extraction. Auto-detects PDF type: "schedule" (splits by Sala/Room sections) vs "book" (RecursiveCharacterTextSplitter). |
| `consumers.py` | `DIDProxyConsumer` — AsyncWebsocketConsumer handling real-time D-ID avatar streaming with LangChain RAG chains. |
| `interview_ai.py` | Gemini 2.0 Flash for interview question generation, answer analysis (feedback + 1-10 rating), and summary generation. Also contains `speech_to_text()` (Gemini multimodal) and `text_to_speech()` (gTTS). |
| `tasks.py` | Celery task `process_pdf_task` — async PDF/CSV processing into FAISS indices. |
| `utils.py` | D-ID API integration: video generation, animation creation, presenter/voice list fetching. |

### External Services

- **Google Gemini** — LLM for Q&A, interview analysis (via `google.generativeai` and `langchain-google-genai`)
- **D-ID** — AI avatar video generation and real-time streaming
- **OpenAI Whisper** — Speech-to-text (used in `VoiceQueryAPIView`)
- **gTTS** — Text-to-speech for interview responses
- **Redis** — Celery broker/backend

### URL Patterns

All API routes are under `/api/` (defined in `Agent_app/urls.py`, included from `Agent_Project/urls.py`). WebSocket route: `ws/stream-avatar/` (defined in `Agent_app/routing.py`).

### Auth

JWT via `rest_framework_simplejwt`. Admin-only endpoints use `IsAdminUser` permission class. Token lifetime: 365 days.

## Important Patterns

- `ASGI_APPLICATION` is set (not WSGI) — the project uses Django Channels for WebSocket support
- `APPEND_SLASH = False` — URLs do not have trailing slashes
- FAISS indices are stored as directories at project root: `{agent_id}_faiss_index/`
- Media uploads go to `media/` (uploads/, interview_resumes/, agent_videos/, agent_images/)
- The `embedder.py` global embedder must be lazy-loaded (`get_embedder()`) because `GoogleGenerativeAIEmbeddings` makes a gRPC connection at init time, which blocks/hangs server startup
- `GOOGLE_API_KEY` env var is required at startup for the embedder import chain to work
- SQLite database — no concurrent write safety
