from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from core.detect import stream, detect

contador = 0

def home(request):
    return render(request, "core/home.html", context={"contador": contador})

def video_feed(request):
    return StreamingHttpResponse(stream(),content_type="multipart/x-mixed-replace;boundary=frame")

def detect_feed(request):
    return StreamingHttpResponse(detect(),content_type="multipart/x-mixed-replace;boundary=frame")