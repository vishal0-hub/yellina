from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate
from .models import Agent, ChatHistory, InterviewQA, InterviewSession, UploadedFile
from django.contrib.auth.models import User


class AdminLoginSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True)
    email = serializers.EmailField(write_only=True)

    def validate(self, data):
        email=data['email']
        #  fetch username
        try:
            print('email--->>>', email)
            usr=User.objects.get(email=email)
            user_name=usr.username
            print("username-->>>>",user_name)
        except Exception as e:
            print(f"exceptio in admin login: {e}")
            raise serializers.ValidationError(" User does not  exists")
        
        user = authenticate(username=user_name, password=data['password'])
        if user and user.is_staff:
            data['user'] = user
            return data
        raise serializers.ValidationError("Invalid credentials or not an admin.")
    
class UploadedFileSerializer(serializers.ModelSerializer):
    file=serializers.SerializerMethodField()
    class Meta:
        model = UploadedFile
        fields = '__all__'
    
    def get_file(self, obj):
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.file.url).replace("http://", "https://")
        return obj.file.url.replace("http://", "https://")


class AgentSummarySerializer(serializers.ModelSerializer):
    UploadedFiles = UploadedFileSerializer(many=True, read_only=True, source='files')
    class Meta:
        model = Agent
        fields = ['id', 'name', 'language', 'voice_tone','voice_id', 'provider','category', "presenter_id","animation_url", "UploadedFiles", "active", 'created_at']


#  api to get the chat
class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = [ "user_message", "ai_response", "created_at"]


class InterviewStartSerializer(serializers.Serializer):
    resume = serializers.FileField()

    def validate_resume(self, value):
        if not value.name.endswith('.pdf'):
            raise serializers.ValidationError("Only PDF files are accepted.")
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size must not exceed 10MB.")
        return value


class InterviewAnswerSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    answer = serializers.CharField(min_length=1, max_length=5000, required=False)
    audio = serializers.CharField(required=False)

    def validate(self, data):
        if not data.get('answer') and not data.get('audio'):
            raise serializers.ValidationError("Either 'answer' (text) or 'audio' (base64) is required.")
        if data.get('answer'):
            data['answer'] = data['answer'].strip()
            if not data['answer']:
                raise serializers.ValidationError({"answer": "Answer cannot be empty."})
        return data


class InterviewQASerializer(serializers.ModelSerializer):
    class Meta:
        model = InterviewQA
        fields = ['question_index', 'question', 'answer', 'feedback', 'rating']