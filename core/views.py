from django.shortcuts import render
from django.http import StreamingHttpResponse
from core.detect import stream, detect

def home(request):
    return render(request, "core/home.html", context={"name": "Teste home"})

def video_feed(request):
    return StreamingHttpResponse(stream(),content_type="multipart/x-mixed-replace;boundary=frame")

def detect_feed(request):
    return StreamingHttpResponse(detect(),content_type="multipart/x-mixed-replace;boundary=frame")