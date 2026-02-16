from django.urls import re_path
from .consumers import DIDProxyConsumer

websocket_urlpatterns = [
    re_path(r"ws/stream-avatar/$", DIDProxyConsumer.as_asgi()),
]
