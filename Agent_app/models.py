from django.db import models


class Agent(models.Model):
    name = models.CharField(max_length=255)
    language = models.CharField(max_length=50)
    voice_tone = models.CharField(max_length=50)
    animation_id = models.CharField(max_length=50, null=True, default=None)
    presenter_id = models.CharField(max_length=50, null=True, default=None)
    animation_url = models.CharField(max_length=200, null=True, default=None)
    provider = models.CharField(max_length=200, null=True, default=None)
    voice_id = models.CharField(max_length=200, null=True, default=None)
    category = models.CharField(max_length=50, null=True, blank=True, default=None)
    gender = models.CharField(max_length=50, null=True, blank=True, default=None)
    active=models.BooleanField(default=False)
    image = models.ImageField(upload_to='agent_images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
class UploadedFile(models.Model):
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='files')
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file.name} for {self.agent.name}"

# DB to  manage the chat history
class ChatHistory(models.Model):
    session_id=models.CharField(max_length=100,default=None, blank=True, null=True)
    user_message=models.TextField()
    ai_response=models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)





