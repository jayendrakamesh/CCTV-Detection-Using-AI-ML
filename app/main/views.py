from django.http import HttpResponse
from django.shortcuts import render, redirect
import cv2 as cv
import numpy as np
from django.http import StreamingHttpResponse
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models

def home(request):
    # Get submitted values
    f1 = request.GET.get("f1", "off")
    f2 = request.GET.get("f2", "off")

    f3 = request.GET.get("f3", "off")
    f4 = request.GET.get("f4", None)

    # Detect available cameras
    cameras = []
    for i in range(10):  
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cameras.append({'index': i, 'resolution': (width, height)})
        cap.release()

    # Pass the data to the template
    context = {
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'cameras': cameras,
    }
    if(f1=="on"):
        movement_feed(request)
    if(f2=="on"):
        door_feed(request)
    if(f3=="on"):
        ppe_feed(request)

    return render(request, 'main/home.html', context)

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to a 'home' page or another URL
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'main/login.html')

import cv2 as cv
import numpy as np
from django.http import StreamingHttpResponse

def resize_with_black_bars(frame, target_width=1280, target_height=720):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_frame = cv.resize(frame, (new_width, new_height))

    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_frame = cv.copyMakeBorder(resized_frame, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    
    return new_frame

def movement_detection():
    cap = cv.VideoCapture("static/testdata/server.mp4")
    ret, framel = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv.absdiff(framel, frame2)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < 900:
                continue
            
            cv.rectangle(framel, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(framel, "Movement Detected", (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                       1, (255, 0, 0), 3)

        framel = resize_with_black_bars(framel)
        ret, buffer = cv.imencode('.jpg', framel)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
        framel = frame2
        ret, frame2 = cap.read()

def door_detection():
    cap = cv.VideoCapture("static/testdata/server.avi")
    ret, framel = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        framel = resize_with_black_bars(framel)
        ret, buffer = cv.imencode('.jpg', framel)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
        framel = frame2
        ret, frame2 = cap.read()

def ppe_detection():
    cap = cv.VideoCapture("static/testdata/PPE.mp4")
    ret, framel = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        framel = resize_with_black_bars(framel)
        ret, buffer = cv.imencode('.jpg', framel)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
        framel = frame2
        ret, frame2 = cap.read()

def movement_feed(request):
    return StreamingHttpResponse(movement_detection(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def door_feed(request):
    return StreamingHttpResponse(door_detection(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def ppe_feed(request):
    return StreamingHttpResponse(ppe_detection(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
