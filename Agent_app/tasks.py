import os
from datetime import datetime, timedelta

from celery import shared_task
from celery.schedules import crontab
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone

from .csv_handler import CSVHandler
from .embedder import Embedder
from .models import Agent, UploadedFile, InterviewSession
from .pdf_handler import PdfHandler


@shared_task
def process_pdf_task(file_id, agent_id):
    try:
        agent = Agent.objects.get(id=agent_id)
        uploaded_file = UploadedFile.objects.get(id=file_id)
        print(f"\n========== [Celery Task Start] ==========")
        print(f"Agent: {agent.name} (ID={agent.id})")
        print(f"File: {uploaded_file.file.name}")

        data = []
        ftype = None
        file_path = uploaded_file.file.path
        book = os.path.basename(uploaded_file.file.name)

        # --- Extract text from PDF or CSV ---
        if uploaded_file.file.name.endswith(".pdf"):
            ftype = "pdf"
            pdf_handler = PdfHandler(file_path)
            full_text = pdf_handler.extract_full_text()

            if not full_text.strip():
                print(f"⚠️ No text found in {book}")
            else:
                for chunk in pdf_handler.chunk_text(full_text):
                    data.append(
                        {"type": "text", "text": chunk, "book": book, "doc_id": book}
                    )

        elif uploaded_file.file.name.endswith(".csv"):
            ftype = "csv"
            csv_handler = CSVHandler(file_path)
            full_text = csv_handler.full_text
            if full_text.strip():
                data.append(
                    {"type": "text", "text": full_text, "book": book, "doc_id": book}
                )
            else:
                print(f"⚠️ CSV {book} has no text content")

        print(f"📄 Extracted {len(data)} text blocks from file")
        if not data:
            print(f"🚫 No valid data extracted from {uploaded_file.file.name}")
            return

        # --- Build FAISS index directory ---
        index_dir = os.path.join(settings.BASE_DIR, f"{agent.id}_faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        print(f"📂 Using index dir: {index_dir}")

        embedder = Embedder(index_path=index_dir, ftype=ftype)

        # --- Create or update FAISS index ---
        index_file_path = os.path.join(index_dir, "index.faiss")
        if os.path.exists(index_file_path):
            print(f"🟡 Existing index found — loading it...")
            embedder.load_faiss_index()
            embedder.add_documents(data)
        else:
            print(f"🆕 Creating a new index for {agent.name}")
            embedder.text_to_docs(data)
            result = embedder.create_faiss_index()
            print(f"🔍 create_faiss_index() result: {result}")

        print(f"✅ Completed FAISS indexing for {agent.name}")
        print(f"========== [Celery Task End] ==========\n")

    except ObjectDoesNotExist:
        print(f"❌ Agent or file not found (agent_id={agent_id}, file_id={file_id})")
    except Exception as e:
        print(f"❌ ERROR: {e}")


@shared_task
def cleanup_expired_interview_sessions():
    """
    Clean up expired interview sessions.
    Runs daily to remove:
    - Incomplete sessions older than 24 hours
    - Completed sessions older than 7 days
    """
    now = timezone.now()
    cleanup_count = 0

    # Delete incomplete sessions older than 24 hours
    incomplete_cutoff = now - timedelta(hours=24)
    incomplete_sessions = InterviewSession.objects.filter(
        status='in_progress',
        created_at__lt=incomplete_cutoff
    )

    for session in incomplete_sessions:
        # Delete associated resume file
        if session.resume_file:
            try:
                os.remove(session.resume_file.path)
            except OSError:
                pass
        cleanup_count += 1

    incomplete_deleted = incomplete_sessions.delete()[0]

    # Delete completed sessions older than 7 days
    completed_cutoff = now - timedelta(days=7)
    completed_sessions = InterviewSession.objects.filter(
        status='completed',
        created_at__lt=completed_cutoff
    )

    for session in completed_sessions:
        # Delete associated resume file
        if session.resume_file:
            try:
                os.remove(session.resume_file.path)
            except OSError:
                pass
        cleanup_count += 1

    completed_deleted = completed_sessions.delete()[0]

    print(f"Interview session cleanup: {incomplete_deleted} incomplete, {completed_deleted} completed sessions removed")
    return f"Cleaned up {cleanup_count} interview sessions"


@shared_task
def mark_abandoned_sessions():
    """
    Mark sessions as abandoned if no activity for 2 hours.
    This runs every hour.
    """
    cutoff_time = timezone.now() - timedelta(hours=2)

    # Find sessions that are still in progress but haven't been updated
    abandoned_sessions = InterviewSession.objects.filter(
        status='in_progress',
        updated_at__lt=cutoff_time
    )

    count = abandoned_sessions.count()
    if count > 0:
        # Update status - but need to add 'abandoned' to model choices first
        for session in abandoned_sessions:
            session.status = 'abandoned'
            session.summary = 'Session was abandoned before completion.'
            session.save()

    print(f"Marked {count} sessions as abandoned")
    return f"Marked {count} sessions as abandoned"
