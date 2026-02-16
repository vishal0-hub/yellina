import asyncio
from datetime import date
import json
import websockets
from channels.generic.websocket import AsyncWebsocketConsumer
import os
from dotenv import load_dotenv
import base64
import time
from .utils import fetch_video_from_did, generate_video_from_did
from urllib.parse import parse_qs
import django
from channels.db import database_sync_to_async
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI
import tempfile
import  requests
import uuid
import httpx

from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
import re
from .embedder import Embedder
from .pdf_handler import PdfHandler
import datetime





os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Agent_Project.settings")
django.setup()
from .models import Agent, UploadedFile, ChatHistory


load_dotenv()

DID_KEY = os.getenv("D_ID_KEY")

# Encode the "username:apikey" string to Base64
did_key_encoded = base64.b64encode(DID_KEY.encode("utf-8")).decode("utf-8")


D_ID_WS_ENDPOINT = "wss://api.d-id.com/talks/streams"

MEDIA_DIR = "media"

#  setup teh consumer for DID Proxy
class DIDProxyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept() 
        self.stream_id=None
        self.conversation_history = []
        self.interview_questions = []
        self.current_question_index = 0
        self.interview_responses = []
        self.is_interview_mode = False
        print('self. converstin history--->>', self.conversation_history)

        

        query_string = self.scope['query_string'].decode()
        query_params = parse_qs(query_string)
        self.agent_id = query_params.get('agent_id', [None])[0]
        self.local_session_id=str(uuid.uuid4())
        # local_session_id = query_params.get('local_session_id', [None])[0]
        # print('local session id--->>',local_session_id)
        # if not local_session_id:
        #     self.local_session_id=str(uuid.uuid4())
        # else:
        #     #  get the previous chat history
        #     self.local_session_id=local_session_id
        #     history=await self.get_chat_history(local_session_id)
        #     self.conversation_history.extend(history)
        #     print('self. converstin history--->>', self.conversation_history)

        print("WebSocket connected with agent_id:", self.agent_id)
        print("WebSocket connection established")


        self.client = OpenAI()


        if not self.agent_id:
            await self.send(text_data=json.dumps({
                "status": "failed",
                "message": f"Please provide the agent_id"
            }))
            await self.close()
            return
        
        # fetch  the agent detail
        agent = await self.get_agent(self.agent_id)
        if not agent:   
            await self.send(text_data=json.dumps({
                "status": "failed",
                "message": f"Agent with id {self.agent_id} not found"
            }))
            await self.close()
            return
        
        # get the agent image url
        image_url = agent.image.url if agent.image else None

        # get the agent language
        self.language = agent.language if agent.language else "english"
        self.voice_id= agent.voice_id if agent.voice_id  else "Sara"
        self.provider= agent.provider if agent.provider  else "microsoft"
        self.voice_tone= agent.voice_tone if agent.voice_tone  else "cheerful"
        self.gender= agent.gender if agent.gender  else "male"
        self.category= agent.category 
        self.session_id=""

        # load index for the agent
        self.embeder=Embedder(index_path=f"{agent.id}_faiss_index")
        res=self.embeder.load_faiss_index()

        # check that agent  type if agent  type is programe

        self.retrived_data=[]

        if  "programm" in  self.category.lower():
             for doc in self.embeder.index.docstore._dict.values():
                  self.retrived_data.append(doc.page_content)
        
        # Check for Interview Mode
        if "interview" in self.category.lower():
            self.is_interview_mode = True
            print("Interview mode detected. Generating questions from resume...")
            resume_text = ""
            if self.embeder.index:
                for doc in self.embeder.index.docstore._dict.values():
                    resume_text += doc.page_content + "\n"
            
            if resume_text:
                self.interview_questions = await self.generate_interview_questions(resume_text)
                print(f"Generated {len(self.interview_questions)} questions.")

        print(self.retrived_data,'\n\n\n----->>>>>>>datat')


        
        self.retriever=self.embeder.index.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        prompt_template = f"""
You are a helpful, precise, and well-mannered {self.gender} Doctor.
You always respond in {self.language}, never in any other language.
You have over 30 years of experience answering medical and academic queries with factual accuracy, linguistic precision, and professional tone.

You have access to the context page_content and the conversation chat_history.
Before answering, you must carefully analyze all provided data and history, verify every name, title, room, date, and topic exactly as they appear in the data, and then align your response with the current date and time.
After complete analysis, provide a clear, accurate, and context-based answer in plain text only (no Markdown, no bullet points, no bold, no document references).

Conversation so far (chat_history):
{{chat_history}}

Context:
{{context}}

Doctor question:
{{question}}

--------------------------
Instructions (follow strictly in this order):

1. Greetings rule:
- If the Doctor greets you and there is no previous chat, greet politely once and briefly.
- If the conversation already has history, do not greet again; focus on providing the answer.
- If the Doctor asks a valid question and the data is available, answer directly and precisely.

2. Context-related rule:
- Use only information explicitly present in the context or verified in the chat history.
- Provide complete, clear, and well-structured answers using the exact wording from the source for names, titles, dates, and rooms.
- Never modify, assume, or simplify names, titles, or terms.
- Clearly distinguish between "Room name" and "Topic/Category" — never confuse or merge them.
- If any detail is missing or unclear, do not guess. State that the information is not available.
- Maintain strict **grammatical agreement** (gender, number, tense) in all titles and sentences.
- Always include required articles and prepositions as shown in the original text (e.g., “un approccio mirato”).
- Preserve correct **capitalization** (e.g., “Sala D”, “Prof. Mario Rossi”).
- When translating titles, provide a faithful and literal translation while keeping the original form and grammatical correctness.

3. Out-of-context rule:
- If the Doctor’s question is unrelated to the context or chat history, reply exactly with:
  "Please ask the question related to the topic only." in {self.language}.
- Then politely suggest asking a question related to the topic.

4. History refinement rule:
- If the Doctor’s question repeats or builds on a previous one, refine and improve the earlier answer by merging verified context data with past replies.
- Do not alter or contradict correct details already confirmed in the history.

5. Accuracy and factual integrity rules:
- Never fabricate, assume, or change any data not explicitly found in the context or chat history.
- Do not invent times, speakers, or session titles.
- In answer don't include any metadata details like image_path, page, book, type, etc. only provide the response from the page_content
- Verify spelling, grammar, and agreement before responding.
- If discrepancies appear between context and history, prefer the most recent and explicitly confirmed data.
- Ensure titles retain correct number (singular/plural) and gender agreement (e.g., “Cataratte complicate” not “Cataratta complicato”).

Answer format rules:
- Respond in plain text, grammatically correct, and coherent.
- Do not use Markdown, bullets, or numbering.
- Keep tone professional, polite, and factual.
- Answer must  include the  datetime, weekday name and location that related to the user query only.
- Always provide concise and relevant answers from the analyzed data.

Final Answer:
"""



        custom_prompt = PromptTemplate(
            input_variables=["chat_history","context", "question"],
            template=prompt_template,
        )

        # Create memory store for conversation
        # self.memory = ConversationBufferWindowMemory(
        #     k=7,
        #     memory_key="chat_history",
        #     return_messages=True
        # )


        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        self.qa_chain = LLMChain(
            llm=llm,
            prompt=custom_prompt,
            # verbose=True
        )

        # send the agent image url to the client
        await self.send(text_data=json.dumps({  
            "status": "success",
            "message": "Agent image URL fetched successfully",
            "image_url": image_url,
            "action": "offer",
            "agent_language": self.language
        }))

    async def disconnect(self, close_code):
        print("WebSocket connection closed")
        # close the stream
        try:
            print('inside disconnect')
            print('self.stream id-->>', self.stream_id)
            # if self.stream_id:
            #     await  self.delete_stream()

        except Exception as e:
            print('exception in  disconect-___>>',e)
        
  
    async def receive(self, text_data):
        data = json.loads(text_data)

        # get the agent id and audio input
        agentId = data.get("agent_id")
        audio_base64 = data.get("audio_input")
        self.session_id = data.get("session_id")
        self.stream_id = data.get("stream_id")
    
        audio_data = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(audio_data)
            audio_path = tmp_audio.name

        # Step 2: Transcribe using OpenAI Whisper API
        try:

            openai_client = OpenAI()  # Reads key from env (OPENAI_API_KEY)
            with open(audio_path, "rb") as file:
                # transcription = openai_client.audio.translations.create(
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file
                )
                user_question = transcription.text.strip()
                print(user_question," question-->>>")

                if self.retrived_data:
                    print('inside retirve dataaa')
                    retrieved_docs=self.retrived_data
                else:
                    # Run query on FAISS index
                    retrieved_docs = self.embeder.query_index(user_question.lower(), k=3)
                    print(f"Retrieved: {retrieved_docs} documents from index for the question.")
                
                print('done-->>>')

                

                # Interview Mode logic
                if self.is_interview_mode:
                    if self.current_question_index < len(self.interview_questions):
                        # Store response for previous question (if not first)
                        if self.current_question_index > 0:
                            self.interview_responses.append({"question": self.interview_questions[self.current_question_index-1], "answer": user_question})
                        
                        # Get next question
                        answer = self.interview_questions[self.current_question_index]
                        self.current_question_index += 1
                        
                        # If session just ended with the last question being answered
                        if self.current_question_index == len(self.interview_questions):
                             # This was the last question being asked, we still need to wait for the final answer
                             pass
                    else:
                        # Final evaluation
                        self.interview_responses.append({"question": self.interview_questions[-1], "answer": user_question})
                        evaluation = await self.evaluate_interview()
                        answer = evaluation
                        self.is_interview_mode = False # End interview
                else:
                    # Run QA (Standard mode)
                    raw_answer = self.qa_chain.run(question=user_question, context = retrieved_docs, chat_history=self.conversation_history, datetime=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
                    print('raw answer--->>>', raw_answer)
                    answer=self.validate_answer(user_question, raw_answer)

                self.conversation_history.append(f"Doctor: {user_question}")
                self.conversation_history.append(f"Assistant: {answer}")
                print("\n\n\nanswer--->>", answer)

                # Save to DB
                await self.save_chat_history(
                    session_id=self.local_session_id,  # fallback if session_id is missing
                    user_message=user_question,
                    ai_response=answer
                )

                # fetch the image path
                image=[]

                # if not self.retrived_data:
                #     for res in retrieved_docs:
                #         if res.metadata.get('image_path',False):
                #             image.append(res.metadata['image_path'])
                #             print("Image path:", res.metadata['image_path'])


        except Exception as e:
            print("Error during speech-to-text or processing:", str(e))
            image=[]
            user_question=''
            if self.language.lower() == 'italian':
                answer="Spiacente, non sono riuscito a elaborare il tuo audio. Per favore riprova."
            else:
                answer="Sorry, I could not process your audio. Please try again."


            # return Response({"error": "Speech-to-text failed", "details": str(e)}, status=500)
        finally:
            os.remove(audio_path)
        
        #  convert text to the time stamp
        # time_stamp=None
        # time_stamp=self.text_to_timed_segments(answer)
        # print('res--->>>', time_stamp)

        if self.stream_id:
            # Now that the stream is active, speak the answer
            video_res=await self.send_utterance_to_did(self.stream_id, answer, agentId)


        # Here you can process the data and send a response
        response = {"message": "Data received successfully", 
                    "user_question":user_question,
                    "answer":answer,
                    "image": image,
                    "status": "done",
                    # "time_stamp": time_stamp,
                    'language': self.language,
                    'local_session_id' : self.local_session_id
                    }
        await self.send(text_data=json.dumps(response))
    
    def validate_answer(self, question: str, answer: str) ->str:
        """
        Validate and  refine the  anwer  provided by  open ai
        """
        try:
            # create the  client to check the  resposne validity  by  open ai
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            validation_prompt = f"""Check the following answer for factual accuracy and relevance to the question asked. You  have to check the answer and  remove the  all the gramatic error, vocabulary issues, and ensure clarity.Some rules to  validate the answer are  given below:

            ## key points:
            - no need  to  add  extra markdown, bullet points, numbering in the answer.
            - no need to  extra keyboard, just provide the  refined answer.

            #### Accuracy and factual integrity rules:
            - Verify spelling, grammar, and agreement before responding.
            - If discrepancies appear between context and history, prefer the most recent and explicitly confirmed data.
            - Ensure titles retain correct number (singular/plural) and gender agreement (e.g., “Cataratte complicate” not “Cataratta complicato”).
            \n\nQuestion: {question}\n\nAnswer: {answer} """

            validation_response = llm.invoke([
                {"role": "system", "content": f"You are  answer validator with  30  years of the experice. you are expert in {self.language}. You  taks is to remove te gramtic error from the answer , ensure the clarity and provide only the  correct answer nothing else."},
                {"role": "user", "content": validation_prompt}
            ])

            f_answer = validation_response.content.strip()
            print("Validation response:", f_answer)
            return f_answer
            
        except Exception as e:
            print('exception in  validate answer-->>', e)
            return answer
    
    def should_remove_page(self, text: str) -> bool:
        """Check if a page should be removed based on first few words."""
        words = text.strip().split()
        if not words:
            return False

        # Normalize (case-insensitive)
        first = words[0].lower()
        second = words[1].lower() if len(words) > 1 else ""
        third = words[2].lower() if len(words) > 2 else ""

        # Rules:
        if first == "prefazione" or first == "abbreviazioni" or first == "sommario" or first == "autori" or first == "introduzione":
            return True
        
        if "elenco autori" in text.lower():
            return True
        
        if first == "indice" or second == "indice":
            return True
        
        # Rule 2: Dot-leader pattern (lots of consecutive dots)
        if re.search(r"\.{5,}", text):  # 5 or more dots in a row
            return True


        return False

    def text_to_timed_segments(self, text, voice="alloy"):
        try:
            """
            Convert text to audio with OpenAI TTS,
            then transcribe it with Whisper to get timestamps.
            Uses a temporary file in a Windows-safe way.
            """
            # Step 1: Create a temp file path (not open yet)
            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()  # close immediately so OpenAI can write to it

            # Generate audio and save to temp file
            with self.client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            ) as response:
                response.stream_to_file(temp_audio_path)

            # Step 2: Transcribe audio with Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # Clean up (delete temp file)
            import os
            os.remove(temp_audio_path)
            # ✅ Convert word objects into plain dicts
            words_as_dicts = [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in transcript.words
            ]

            return words_as_dicts

        except Exception as e:
            print('exception in convert text to timestamp', e)
            return None


    #start did stream
    async def start_did_stream(self, source_url="https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg"):
        url = "https://api.d-id.com/talks/streams"

        payload = {
            "stream_warmup": "false",
            "source_url": source_url
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }

        response = requests.post(url, json=payload, headers=headers)

        print(response.text)
        return response.json()
    

    async def send_answer_to_did(self, stream_id, sdp):
        url = f"https://api.d-id.com/talks/streams/{stream_id}/sdp"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }
        payload = {
            "sdp": sdp
        }

        response = requests.post(url, json=payload, headers=headers)
        print("SDP Answer Response:", response.status_code)
        print("SDP Answer Response:", response.json())

    async def send_ice_candidate(self, candidate):
        url = f"https://api.d-id.com/talks/streams/{self.stream_id}/ice"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }
        payload = { "candidate": candidate }  # candidate can be null
        response = requests.post(url, json=payload, headers=headers)
        print("ICE Candidate Response:", response.status_code)

    #  setting the  agent stream
    async def send_utterance_to_did(self, stream_id, text, agentId):
        url = f"https://api.d-id.com/clips/streams/{stream_id}"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Basic {did_key_encoded}"
        }


        payload = {
            "script": {
                "type": "text",
                "provider": {
                    "type": self.provider,
                    "voice_config": {
                            "style": self.voice_tone,
                        },
                    "voice_id": self.voice_id
                },
                "input": text,
                "ssml": "false",
            },
            "config": { "result_format": "mp4" },
            "session_id": self.session_id

        }

        response = requests.post(url, json=payload, headers=headers)
        print("Utterance Response:", response.status_code)
        print("Utterance Response:", response.json())
        return response.json()
    

    # async def delete_stream(self):
    #     url = f"https://api.d-id.com/talks/streams/{self.stream_id}"
    #     headers = {
    #         "authorization": f"Basic {did_key_encoded}",
    #         "accept": "application/json"
    #     }
    #     response = requests.delete(url, headers=headers)
    #     print("Stream Deleted:", response.status_code)
    #     self.stream_id = None

    async def delete_stream(self):
        if not getattr(self, "stream_id", None):
            return

        url = f"https://api.d-id.com/talks/streams/{self.stream_id}"
        headers = {
            "authorization": f"Basic {did_key_encoded}",
            "accept": "application/json"
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.delete(url, headers=headers)
                print("Stream Deleted:", response.status_code)
            except Exception as e:
                print("Error deleting stream:", e)

        self.stream_id = None


    @database_sync_to_async
    def get_agent(self, agent_id):
        return Agent.objects.filter(id=agent_id).first()

        
    
    #  load file for the agent
    @database_sync_to_async
    def get_uploaded_files(self, agent_id):
        print("Fetching uploaded files for agent_id:", agent_id)
        data= list(UploadedFile.objects.filter(agent_id=agent_id).order_by('-uploaded_at'))
        print("Fetched files:", data)   
        return data

    @database_sync_to_async
    def save_chat_history(self, session_id, user_message, ai_response):
        ChatHistory.objects.create(
            session_id=session_id,
            user_message=user_message,
            ai_response=ai_response
        )
    
    @database_sync_to_async
    def get_chat_history(self, session_id):
        history_data=ChatHistory.objects.filter(
            session_id=session_id,
        ).values('user_message', 'ai_response', 'created_at').order_by('created_at')

        print('history_data-->>', history_data)

        # convert to list
        history_list=list(history_data)[:5]
        print('history list->>>', history_list)

        # convert to the converstion format
        history_list=[f"Doctor: {item['user_message']}\nAssistant: {item['ai_response']}" for item in history_list]

        return  history_list
    
    #  for generate imaeg for the question
    def generate_image(self,question:str =None):
        try:
            if question:
                print('genrting image')
                prompt=f"You are  great diagram  generator. User will pass the  question  to  you  and  you  generate the images for the  provided question   so  user will easily understand. Here is the user question:{question}"
                #  call api to generate image
                img = self.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    # size="1024x1024",
                    # response_format="url"
                )

                # image_bytes = base64.b64decode(img.data[0].b64_json)
                print('image url:', img)
                img_url=img.data[0].url
                print('image generated')
                return img_url
            

            return None
        except Exception as e:
            print('exception in  generate image:', e)
            return None

    async def generate_interview_questions(self, resume_text):
        """Generates 5 interview questions based on the resume."""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            prompt = f"Based on the following resume text, generate 5 technical and behavioral interview questions. Return ONLY the questions as a JSON list of strings.\n\nResume:\n{resume_text}"
            response = await llm.ainvoke(prompt)
            # Find the JSON part
            content = response.content
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                return questions
            return ["Tell me about yourself.", "What are your core strengths?", "Describe a difficult project you worked on.", "Where do you see yourself in 5 years?", "Why should we hire you?"]
        except Exception as e:
            print(f"Error generating questions: {e}")
            return ["Tell me about yourself.", "What are your core strengths?", "Describe a difficult project you worked on.", "Where do you see yourself in 5 years?", "Why should we hire you?"]

    async def evaluate_interview(self):
        """Evaluates the interview based on responses."""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            chat_summary = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in self.interview_responses])
            prompt = f"Analyze the following interview session and provide a rating and accuracy analysis for the candidate's answers. Be professional and constructive.\n\nSession:\n{chat_summary}"
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error evaluating interview: {e}")
            return "Thank you for the interview. We will analyze your responses and get back to you with a rating."
