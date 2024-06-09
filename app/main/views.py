from django.http import HttpResponse
from django.shortcuts import render
import cv2 as cv
import numpy as np
from django.http import StreamingHttpResponse

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
        video_feed(request)

    return render(request, 'main/home.html', context)

def movement_detection(context):
    cap = cv.VideoCapture(0)
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

        ret, buffer = cv.imencode('.jpg', framel)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')
        framel = frame2
        ret, frame2 = cap.read()

def video_feed(request):
    return StreamingHttpResponse(movement_detection({}),
                                 content_type='multipart/x-mixed-replace; boundary=frame')