"""
ASGI config for Agent_Project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import django
from django.core.asgi import get_asgi_application
import Agent_app.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Agent_Project.settings')

# application = get_asgi_application()

application=ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            Agent_app.routing.websocket_urlpatterns
        )
    ),
})
