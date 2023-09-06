from django.contrib import admin
from django.urls import path
from core.views import home
from core.views import video_feed, detect_feed, class_feed

urlpatterns = [
    path("", home, name="home"),
    path('video_feed', video_feed, name='video_feed'),
    path('detect_feed', detect_feed, name='detect_feed'),
    path('class_feed', class_feed, name='class_feed'),
]
