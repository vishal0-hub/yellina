import ast
import base64
import os
import re
import shutil
import tempfile
import time

import requests
from django.conf import settings
from django.db import transaction
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import OpenAI
from rest_framework import status
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .csv_handler import CSVHandler
from .embedder import Embedder
from .models import Agent, ChatHistory, UploadedFile
from .pdf_handler import PdfHandler
from .serializers import (
    AdminLoginSerializer,
    AgentSummarySerializer,
    ChatHistorySerializer,
    UploadedFileSerializer,
)
from .tasks import process_pdf_task
from .utils import (
    create_animation,
    fetch_video_from_did,
    generate_video_from_did,
    get_presenter_list,
    get_voice_list,
)

load_dotenv()
client = OpenAI(api_key="OPENAI_API_KEY")
MEDIA_DIR = "media"


class AdminLoginView(APIView):
    def post(self, request):
        serializer = AdminLoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data["user"]
            refresh = RefreshToken.for_user(user)
            return Response(
                {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                    "username": user.username,
                    "is_staff": user.is_staff,
                }
            )
        return Response(
            {"message": "Unable to login", "error": serializer.errors},
            status=status.HTTP_401_UNAUTHORIZED,
        )


# class FileUploadView(APIView):
# permission_classes=[IsAdminUser]
# def post(self, request, *args, **kwargs):
#     files = request.FILES.getlist('files')
#     image = request.FILES.get('image')
#     presenter_id = request.data.get('presenter_id')
#     animation_url = request.data.get('idle_video')
#     agent_names = request.data.get('agent_name')
#     languages = request.data.get('language')
#     voice_tones = request.data.get('voice_tone')
#     category = request.data.get('category')
#     gender = request.data.get('gender')
#     provider = request.data.get('provider')
#     voice_id = request.data.get('voice_id')

#     print("Agent name:", agent_names)
#     print("Language:", languages)
#     print("Voice tone:", voice_tones)
#     print("Presenter ID:", presenter_id)
#     print("category:", category)
#     print("Animation URL:", animation_url)
#     print("Provider:", provider)
#     print("Voice ID:", voice_id)
#     print("Gender ID:", gender)


#     if not files:
#         return Response({"message": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)
#     if not agent_names or not languages or not voice_tones or not category or not presenter_id or not animation_url or not gender:
#         return Response({"message": "Agent name, language, voice tone, Presenter ID, Gender, and Animation URL are required."}, status=status.HTTP_400_BAD_REQUEST)

#     if gender.lower() not in ['male', "female"]:
#         return Response({"message": "Invalid gender. Please select either 'male' or 'female'."}, status=status.HTTP_400_BAD_REQUEST)

#     try:
#         with transaction.atomic():
#             # Create the agent
#             agent = Agent.objects.create(
#                 name=agent_names,
#                 language=languages,
#                 voice_tone=voice_tones,
#                 presenter_id=presenter_id,
#                 animation_url=animation_url,
#                 image=image if image else None,
#                 category=category,
#                 provider=provider,
#                 voice_id=voice_id,
#                 gender=gender
#             )

#             uploaded_files = []
#             for i, file in enumerate(files):
#                 uploaded = UploadedFile.objects.create(
#                     file=file,
#                     agent=agent,
#                 )
#                 uploaded_files.append(uploaded)

#             print("Uploaded files:", uploaded_files)

#             serializer = UploadedFileSerializer(uploaded_files, many=True)

#             agent.save()
#             print("Agent created with ID:", agent.id)

#             # # Combine image data and text chunks
#             data = []

#             for file in uploaded_files:
#                 ftype="pdf"
#                 if file.file.name.endswith('.pdf'):
#                     pdf_path = file.file.path
#                     book = file.file.name.split("/")[-1]
#                     print("book with  create--->>",book)
#                     print("Processing PDF:", pdf_path)
#                     pdf_handler = PdfHandler(pdf_path)
#                     # image_data = pdf_handler.extract_images(output_folder=f"{MEDIA_DIR}/extracted_images/{agent.id}")
#                     full_text = pdf_handler.extract_full_text()
#                     # print(f"Extracted {len(image_data)} images and texts from {file.file.name}")
#                     print(f"Extracted full text length: {len(full_text)} characters")

#                     # # Add image entries
#                     # for img in image_data:
#                     #     data.append({
#                     #         "type": "image",
#                     #         "text": img["text"],   # image context
#                     #         "image_path": img["image_path"],
#                     #         "page": img["page"],
#                     #         "book": book,
#                     #         "doc_id": book,
#                     #     })

#                     # # Add text entries
#                     for chunk in pdf_handler.chunk_text(full_text):
#                         # room_match = re.search(r'(Sala|Room)\s+[A-Za-z]', chunk, re.IGNORECASE)
#                         # room_name = room_match.group(0).lower() if room_match else None
#                         # meta = pdf_handler.extract_metadata(chunk)
#                         data.append({
#                             "type": "text",
#                             "text": chunk,
#                             "image_path": None,
#                             "page": None,
#                             "book": book,
#                             "doc_id": book,
#                             # "room": meta.get("room"),
#                             # "date": meta.get("date"),
#                             # "time": meta.get("time"),

#                         })
#                 elif file.file.name.endswith('.csv'):
#                     ftype="csv"
#                     csv_path = file.file.path
#                     book = file.file.name.split("/")[-1]
#                     print("book with  create--->>",book)
#                     print("Processing CSV:", csv_path)
#                     csv_handler = CSVHandler(csv_path)
#                     full_text = csv_handler.full_text
#                     if full_text is not None:
#                         # Convert DataFrame to string for chunking
#                         # csv_text = full_text.to_string(index=False)
#                         csv_text = full_text

#                         data.append({
#                                 "type": "text",
#                                 "text": csv_text,
#                                 "image_path": None,
#                                 "page": None,
#                                 "book": book,
#                                 "doc_id": book,
#                             })


#             # Here you can save or process the extracted data as needed
#             embedder = Embedder(index_path=f"{agent.id}_faiss_index", ftype=ftype)

#             #  convert text to documents
#             res=embedder.text_to_docs(data)
#             if res:
#                 # create faiss index
#                 embedder.create_faiss_index()

#             # return Response(serializer.data, status=status.HTTP_201_CREATED)
#             return Response({
#                 "agent_id": agent.id,
#                 "agent_name": agent.name,
#                 "category": agent.category,
#                 "language": agent.language,
#                 "avatar": agent.image.url if agent.image else None,
#                 "uploaded_files": serializer.data
#             }, status=status.HTTP_201_CREATED)

#     except ValueError as ve:
#         print(f"ValueError: {str(ve)}")
#         return Response({"message": str(ve)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         return Response({"message": "Failed to create agent due to an unexpected error."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# def patch(self, request, agent_id, *args, **kwargs):
#     """Partial update of agent (and optionally files)"""
#     try:
#         agent = Agent.objects.get(id=agent_id)
#     except Agent.DoesNotExist:
#         return Response({"message": "Agent not found."}, status=status.HTTP_404_NOT_FOUND)

#     files = request.FILES.getlist('files')
#     image = request.FILES.get('image')

#     # file to delete
#     files_to_remove = request.data.get("files_to_remove", [])
#     if files_to_remove:
#         files_to_remove = ast.literal_eval(files_to_remove)
#         if files_to_remove:
#             # for fid in files_to_remove:
#             UploadedFile.objects.filter(id__in=files_to_remove, agent=agent_id).delete()

#     # Update only fields that are provided
#     agent.name = request.data.get('agent_name', agent.name)
#     agent.language = request.data.get('language', agent.language)
#     agent.voice_tone = request.data.get('voice_tone', agent.voice_tone)
#     agent.presenter_id = request.data.get('presenter_id', agent.presenter_id)
#     agent.animation_url = request.data.get('idle_video', agent.animation_url)
#     agent.category = request.data.get('category', agent.category)
#     agent.provider = request.data.get('provider', agent.provider)
#     agent.voice_id = request.data.get('voice_id', agent.voice_id)
#     agent.gender = request.data.get('gender', agent.gender)

#     if image:
#         agent.image = image

#     try:
#         with transaction.atomic():
#             agent.save()

#             uploaded_files = []
#             if files:
#                 for file in files:
#                     uploaded = UploadedFile.objects.create(file=file, agent=agent)
#                     uploaded_files.append(uploaded)

#             serializer = UploadedFileSerializer(uploaded_files, many=True)


#             if uploaded_files:
#                 print("FAISS index loaded successfully for update.")
#                 # load the index
#                 embedder=Embedder(index_path=f"{agent.id}_faiss_index")
#                 res=embedder.load_faiss_index()
#                 #  update the embder to for te new document
#                 data = []
#                 if res:
#                     for file in uploaded_files:
#                         if file.file.name.endswith('.pdf'):
#                             book = file.file.name.split("/")[-1]
#                             print("book in update--->>",book)
#                             pdf_path = file.file.path
#                             print("Processing PDF for update:", pdf_path)
#                             pdf_handler = PdfHandler(pdf_path)
#                             # image_data = pdf_handler.extract_images(output_folder=f"{MEDIA_DIR}/extracted_images/{agent.id}")
#                             full_text = pdf_handler.extract_full_text()
#                             # print(f"Extracted {len(image_data)} images and texts from {file.file.name}")
#                             print(f"Extracted full text length: {len(full_text)} characters")

#                             # # # Add image entries
#                             # for img in image_data:
#                             #     data.append({
#                             #         "type": "image",
#                             #         "text": img["text"],   # image context
#                             #         "image_path": img["image_path"],
#                             #         "page": img["page"],
#                             #         "book": book,
#                             #         "doc_id": book,
#                             #     })

#                             # # Add text entries
#                             for chunk in pdf_handler.chunk_text(full_text):
#                                 # meta = pdf_handler.extract_metadata(chunk)
#                                 data.append({
#                                     "type": "text",
#                                     "text": chunk,
#                                     "image_path": None,
#                                     "page": None,
#                                     "book": book,
#                                     "doc_id": book,
#                                     # "room": meta.get("room"),
#                                 })

#                     #  add new document
#                     embedder.add_documents(data)

#             return Response({
#                 "agent_id": agent.id,
#                 "agent_name": agent.name,
#                 "category": agent.category,
#                 "language": agent.language,
#                 "avatar": agent.image.url if agent.image else None,
#                 "uploaded_files": serializer.data if files else "No new files uploaded"
#             }, status=status.HTTP_200_OK)

#     except Exception as e:
#         return Response({"message": f"Failed to update agent: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FileUploadView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist("files")
        image = request.FILES.get("image")
        presenter_id = request.data.get("presenter_id")
        animation_url = request.data.get("idle_video")
        agent_names = request.data.get("agent_name")
        languages = request.data.get("language")
        voice_tones = request.data.get("voice_tone")
        category = request.data.get("category")
        gender = request.data.get("gender")
        provider = request.data.get("provider")
        voice_id = request.data.get("voice_id")

        print("🔹 Incoming Data:")
        print(f"Agent Name: {agent_names}")
        print(f"Language: {languages}")
        print(f"Voice Tone: {voice_tones}")
        print(f"Presenter ID: {presenter_id}")
        print(f"Category: {category}")
        print(f"Animation URL: {animation_url}")
        print(f"Gender: {gender}")
        print(f"Provider: {provider}")
        print(f"Voice ID: {voice_id}")

        if not files:
            print("❌ No files uploaded.")
            return Response(
                {"message": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST
            )

        required = [
            agent_names,
            languages,
            voice_tones,
            category,
            presenter_id,
            animation_url,
            gender,
        ]
        if not all(required):
            print("❌ Missing required fields.")
            return Response(
                {"message": "Missing required fields."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if gender.lower() not in ["male", "female"]:
            print("❌ Invalid gender.")
            return Response(
                {"message": "Invalid gender. Use 'male' or 'female'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            with transaction.atomic():
                agent = Agent.objects.create(
                    name=agent_names,
                    language=languages,
                    voice_tone=voice_tones,
                    presenter_id=presenter_id,
                    animation_url=animation_url,
                    image=image if image else None,
                    category=category,
                    provider=provider,
                    voice_id=voice_id,
                    gender=gender,
                )

                uploaded_files = []
                for file in files:
                    uploaded = UploadedFile.objects.create(file=file, agent=agent)
                    uploaded_files.append(uploaded)

                serializer = UploadedFileSerializer(uploaded_files, many=True)
                agent.save()

                print(
                    f"✅ Agent created successfully → ID: {agent.id}, Name: {agent.name}"
                )
                print("📤 Sending background tasks to Celery...")

                for uploaded_file in uploaded_files:
                    process_pdf_task.delay(uploaded_file.id, agent.id)
                    print(f"➡️ Task queued for file: {uploaded_file.file.name}")

                response_data = {
                    "message": "Agent created successfully. Files are being processed in the background.",
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "category": agent.category,
                    "language": agent.language,
                    "avatar": agent.image.url if agent.image else None,
                    "uploaded_files": serializer.data,
                }

                print("✅ Final Response:")
                print(response_data)

                return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            return Response(
                {"message": "Failed to create agent."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # PATCH for updating agent
    def patch(self, request, agent_id, *args, **kwargs):
        print(f"🔄 Updating Agent ID: {agent_id}")
        try:
            agent = Agent.objects.get(id=agent_id)
        except Agent.DoesNotExist:
            print("❌ Agent not found.")
            return Response(
                {"message": "Agent not found."}, status=status.HTTP_404_NOT_FOUND
            )

        files = request.FILES.getlist("files")
        image = request.FILES.get("image")

        files_to_remove = request.data.get("files_to_remove", [])
        if files_to_remove:
            try:
                files_to_remove = ast.literal_eval(files_to_remove)
                UploadedFile.objects.filter(
                    id__in=files_to_remove, agent=agent
                ).delete()
                print(f"🗑️ Deleted files with IDs: {files_to_remove}")
            except Exception as e:
                print(f"⚠️ File removal parse error: {e}")

        # Update provided fields
        for field, key in [
            ("name", "agent_name"),
            ("language", "language"),
            ("voice_tone", "voice_tone"),
            ("presenter_id", "presenter_id"),
            ("animation_url", "idle_video"),
            ("category", "category"),
            ("provider", "provider"),
            ("voice_id", "voice_id"),
            ("gender", "gender"),
        ]:
            new_value = request.data.get(key)
            if new_value:
                setattr(agent, field, new_value)
                print(f"✏️ Updated {field}: {new_value}")

        if image:
            agent.image = image
            print("🖼️ Updated agent image.")

        try:
            with transaction.atomic():
                agent.save()
                print(f"✅ Agent updated successfully: {agent.id}")

                uploaded_files = []
                if files:
                    for file in files:
                        uploaded = UploadedFile.objects.create(file=file, agent=agent)
                        uploaded_files.append(uploaded)
                        print(f"📎 New file added: {file.name}")
                        process_pdf_task.delay(uploaded.id, agent.id)
                        print(f"➡️ Celery task queued for file: {file.name}")

                serializer = UploadedFileSerializer(uploaded_files, many=True)

                response_data = {
                    "message": "Agent updated successfully.",
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "category": agent.category,
                    "language": agent.language,
                    "avatar": agent.image.url if agent.image else None,
                    "uploaded_files": serializer.data
                    if files
                    else "No new files uploaded",
                }

                print("✅ Final PATCH Response:")
                print(response_data)

                return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"❌ Error updating agent: {e}")
            return Response(
                {"message": f"Failed to update agent: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AgentListView(APIView):
    def get(self, request):
        agents = Agent.objects.all().order_by("-created_at")
        serializer = AgentSummarySerializer(
            agents, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_200_OK)


class VoiceQueryAPIView(APIView):
    def post(self, request):
        # Step 1: Decode base64 audio
        audio_base64 = request.data.get("audio")
        agent_id = request.data.get("agent_id", None)

        if not audio_base64:
            return Response(
                {"message": "Audio not provided"}, status=status.HTTP_400_BAD_REQUEST
            )
        if not agent_id:
            return Response(
                {"message": "Agent ID not provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        audio_data = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(audio_data)
            audio_path = tmp_audio.name

        # Step 2: Transcribe using OpenAI Whisper API
        try:
            openai_client = OpenAI()  # Reads key from env (OPENAI_API_KEY)
            with open(audio_path, "rb") as file:
                transcription = openai_client.audio.translations.create(
                    model="whisper-1", file=file
                )
                user_question = transcription.text.strip()
                print(user_question, " question-->>>")
        except Exception as e:
            return Response(
                {"message": "Speech-to-text failed", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            os.remove(audio_path)

        # Step 3: Load the latest uploaded PDF
        # load pdf  file for agent_id
        file_obj = UploadedFile.objects.filter(agent_id=agent_id)

        print(file_obj, "file_obj--->>>")

        #  if not file_obj.exists():
        if not file_obj:
            return Response(
                {"message": "No files found for the specified agent"},
                status=status.HTTP_404_NOT_FOUND,
            )

        #  read the   file path
        for file in file_obj:
            print(file.file.path, "file.file.path--->>>")
            if file.file.name.endswith(".pdf"):
                latest_pdf = file.file.path
                break

        print(latest_pdf, "latest_pdf--->>>")

        # Step 4: LangChain PDF QA
        try:
            # Load and split PDF
            loader = PyPDFLoader(latest_pdf)
            pages = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(pages)

            # Embedding and FAISS vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = FAISS.from_documents(texts, embeddings)

            print(vectordb.as_retriever(), "vectordb.as_retriever()--->>>")

            # Use Gemini for LangChain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

            # RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever(),
                return_source_documents=False,
            )

            # Run QA
            answer = qa_chain.run(user_question)
            if "I don't know" in answer or not answer.strip():
                answer = "Please ask from the PDF only."

            #  Generate video using D-ID
            generated_data = generate_video_from_did(text=answer)

            while True:
                if generated_data.get("status") == "created":
                    print("Video is being processed...")
                    time.sleep(5)  # Wait for a while before checking again
                    generated_data = fetch_video_from_did(generated_data.get("id"))
                else:
                    print("Video processing completed.")
                    break

            # fetch the id of the video
            video_id = generated_data.get("id")
            print(video_id, "generated_data")

            # Fetch the video URL
            video_data = fetch_video_from_did(video_id)
            video_url = video_data.get("result_url")
            print(video_url, "video_url")

        except Exception as e:
            return Response(
                {"message": "PDF QA failed", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {"question": user_question, "answer": answer, "video_url": video_url},
            status=status.HTTP_200_OK,
        )


#  api for activate or  deactive the agent
class UpdateAgentStatusView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        agent_id = request.data.get("agent_id")

        if not agent_id:
            return Response(
                {"message": "Agent Id is  required"}, status=status.HTTP_400_BAD_REQUEST
            )

        #  fetch id
        try:
            agent_obj = Agent.objects.get(id=agent_id)

        except Agent.DoesNotExist:
            return Response(
                {"message": "Agent ID is invalid."}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"message": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        #  check  that if try to activate the new agent then  deactivate teh previous  agent of the same category
        if not agent_obj.active:
            #  fetch if there another bot is active
            active_agents = Agent.objects.filter(
                active=True, category=agent_obj.category
            )

            print("active_agents", active_agents)
            if active_agents.exists():
                #  deactivate the previous agent
                active_agents.update(active=False)

        agent_obj.active = not agent_obj.active
        agent_obj.save()

        print("agen active", agent_obj.active)

        status_label = "Actived" if agent_obj.active else "Deactivated"
        print("Agent status  updated succesfully")

        return Response(
            {"message": f"Agent '{agent_obj.name}' {status_label} successfully."},
            status=status.HTTP_200_OK,
        )


#  ap to delete the  agent
class DeleAgentView(APIView):
    permission_classes = [IsAdminUser]

    def delete(self, request, agent_id):
        # agent_id=request.data.get("agent_id")

        if not agent_id:
            return Response(
                {"message": "Agent Id is  required"}, status=status.HTTP_400_BAD_REQUEST
            )

        #  fetch id
        try:
            agent_obj = Agent.objects.get(id=agent_id)

        except Agent.DoesNotExist:
            return Response(
                {"message": "Agent ID is invalid."}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"message": "An unexpected error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # delete agent
        agent_obj.delete()

        #  delete related index
        index_path = f"{agent_id}_faiss_index"
        if os.path.exists(index_path):
            if os.path.isdir(index_path):
                shutil.rmtree(index_path)
            else:
                os.remove(index_path)

        # delete extracted images folder for this agent
        media_path = os.path.join(MEDIA_DIR, "extracted_images", str(agent_id))
        if os.path.exists(media_path):
            shutil.rmtree(media_path)

        print("Agent deleted successfully")

        return Response(
            {"message": f"Agent deleted successfully."}, status=status.HTTP_200_OK
        )


#  api  to  get the presenter list from the D-ID
class GetPresenterListView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        # Fetch the presenter list from D-ID
        try:
            presenters = get_presenter_list()
            return Response({"presenters": presenters}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"message": "Presenter list not  fetched", "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# api to get the voice list
class GetVoiceListView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        gender = request.query_params.get("gender", "male")
        language = request.query_params.get("language", "english")
        print("Fetching voice list with gender:", gender, "and language:", language)

        # Fetch the voice list from D-ID
        try:
            voices = get_voice_list(gender=gender.lower(), language=language.lower())
            return Response({"voices": voices}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


#  webhook url
class WebhookView(APIView):
    def post(self, request):
        result_url = request.data.get("result_url")
        animation_id = request.data.get("id")

        print("data-->>>", result_url)
        print("data-->>>", animation_id)

        # fetch the agent wth the animation ID
        try:
            agent = Agent.objects.get(animation_id=animation_id)

            # Create a folder for videos if it doesn't exist
            video_dir = os.path.join(settings.MEDIA_ROOT, "agent_videos")
            os.makedirs(video_dir, exist_ok=True)

            # Create a file path for saving
            file_name = f"{animation_id}.mp4"  # or parse from URL if needed
            file_path = os.path.join(video_dir, file_name)

            # Download the file
            response = requests.get(result_url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as video_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        video_file.write(chunk)

            # Save relative path in model
            relative_path = os.path.join("agent_videos", file_name)
            media_url_path = f"{settings.MEDIA_URL}{relative_path}"
            agent.animation_url = media_url_path  # If FileField, can directly assign
            agent.save()

            # agent.animation_url=result_url
            # agent.save()
            return Response({"message": "success"}, status=status.HTTP_200_OK)
        except Exception as e:
            print("exception in webhook-->>>", e)
            return Response(
                {"message": "Animation  Id not fetched"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ChatHistoryView(APIView):
    """
    API to fetch chat history for a given session_id
    """

    def get(self, request, session_id):
        try:
            chats = ChatHistory.objects.filter(session_id=session_id).order_by(
                "created_at"
            )
            serializer = ChatHistorySerializer(chats, many=True)
            return Response(
                {"session_id": session_id, "history": serializer.data},
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
