from django.contrib import admin
from .models import Agent, UploadedFile

# Register your models here.
admin.site.register(Agent)
admin.site.register(UploadedFile)