from django.urls import path
from . import views

urlpatterns = [
    path('admin-login', views.AdminLoginView.as_view(), name='admin-login'),
    path('upload', views.FileUploadView.as_view(), name='upload'),
    # Partial update agent (PATCH)
    path('upload/<int:agent_id>', views.FileUploadView.as_view(), name='upload-update'),
    path('Agent', views.AgentListView.as_view(), name='Agent'),
    path('Voice', views.VoiceQueryAPIView.as_view(), name='voice-query'),
    path('UpdateAgent', views.UpdateAgentStatusView.as_view(), name='update-agent-status'),
    path('Delete/<int:agent_id>', views.DeleAgentView.as_view(), name='delete-agent-status'),
    path('get_presenter', views.GetPresenterListView.as_view(), name='get-presenter-list'),
    path('get_voice', views.GetVoiceListView.as_view(), name='get-voice-list'),
    path('Webhook', views.WebhookView.as_view(), name='webhook'),
    path("chat-history/<str:session_id>", views.ChatHistoryView.as_view(), name="chat_history"),  
]
