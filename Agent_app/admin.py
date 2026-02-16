from django.contrib import admin
from .models import Agent, InterviewQA, InterviewSession, UploadedFile

# Register your models here.
admin.site.register(Agent)
admin.site.register(UploadedFile)
admin.site.register(InterviewSession)
admin.site.register(InterviewQA)