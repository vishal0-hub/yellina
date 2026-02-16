import uuid

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


class InterviewSession(models.Model):
    STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ]

    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='interview_sessions',
        help_text='User who created this interview session (if authenticated)'
    )
    resume_file = models.FileField(upload_to='interview_resumes/')
    resume_text = models.TextField(blank=True, default='')
    questions = models.JSONField(default=list)
    current_question_index = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='in_progress')
    summary = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Interview {self.session_id} - {self.status}"

    @property
    def total_questions(self):
        return len(self.questions)

    @property
    def is_completed(self):
        return self.status == 'completed'


class InterviewQA(models.Model):
    session = models.ForeignKey(
        InterviewSession,
        on_delete=models.CASCADE,
        related_name='qa_pairs',
    )
    question_index = models.IntegerField()
    question = models.TextField()
    answer = models.TextField(blank=True, default='')
    feedback = models.TextField(blank=True, default='')
    rating = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['question_index']
        unique_together = ['session', 'question_index']

    def __str__(self):
        return f"Q{self.question_index + 1} for {self.session.session_id}"

