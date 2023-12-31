from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from core.detect import stream, detect, opencv_contador_padrao

def home(request):
    return render(request, "core/home.html")

def video_feed(request):
    return StreamingHttpResponse(stream(),content_type="multipart/x-mixed-replace;boundary=frame")

def detect_feed(request):
    return StreamingHttpResponse(detect(),content_type="multipart/x-mixed-replace;boundary=frame")

def contador_padrao(request):
    return StreamingHttpResponse(opencv_contador_padrao(),content_type="multipart/x-mixed-replace;boundary=frame")