from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import cv2 as cv
import torch
from ultralytics import YOLO
from django.utils import timezone
from .models import DoorDetection
from django.core.files.base import ContentFile
from io import BytesIO
from PIL import Image
from django.http import Http404, HttpResponse
import time
from django.core.mail import send_mail
from django.utils import timezone
from django.conf import settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model1 = YOLO('main/best.pt').to(device)
model2 = YOLO('main/best2.pt').to(device)
last_save_time = 0

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'main/login.html')

def home(request):

    request.session['f1'] = request.GET.get("f1", 'off')
    request.session['f2'] = request.GET.get("f2", 'off')
    request.session['f3'] = request.GET.get("f3", 'off')
    request.session['f4'] = request.GET.get("f4", 'off')

    if request.session['f4'] == 'on':
        # Get all detections from the database
        all_detections = DoorDetection.objects.all()

        # Prepare email content
        if all_detections.exists():
            message = "All detections:\n\n"
            for detection in all_detections:
                message += f"Time: {detection.detection_time.strftime('%Y-%m-%d %H:%M:%S')}, Label: {detection.result}\n"
        else:
            message = "No detections recorded."

    # Send the email to the logged-in user
        send_mail(
            subject="All Detections Report",
            message=message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[request.user.email],
            fail_silently=False,
        )

    cameras = []
    for i in range(3):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cameras.append({'index': i, 'resolution': (width, height)})
        cap.release()

    context = {
        'f1': request.session['f1'],
        'f2': request.session['f2'],
        'f3': request.session['f3'],
        'cameras': cameras,
    }

    return render(request, 'main/home.html', context)

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

def generate_camera_stream(camera_index, apply_detection1, apply_detection2):
    cap = cv.VideoCapture(camera_index)
    class_names1 = ['Closed Door', 'Open door', 'Human Action']
    class_names2 = ['boots', 'earmuffs', 'glasses', 'gloves', 'helmet', 'person', 'vest']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if apply_detection1:
            results = model1.predict(source=frame, conf=0.25, save=False, show=False, device=device, half=True)
            frame = results[0].plot()  # Draw detection results on the frame

            for result in results[0].boxes:
                class_index = int(results[0].boxes.cls[0].item())
                if class_names1[class_index] == 'Open door':  # Adjust this based on the actual label/index used in your model
                    save_detection(frame, 'Open Door')
        
        if apply_detection2:
            results = model2.predict(source=frame, conf=0.25, save=False, show=False, device=device, half=True)
            frame = results[0].plot()  # Draw detection results on the frame
                
        # Encode the frame as JPEG
        ret, jpeg = cv.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
    cap.release()

@login_required
def stream_camera(request, camera_index):
    # Check if f1 is 'on' to apply door detection
    choice1 = False
    choice2 = False
    if request.GET.get("f1", request.session.get('f1', 'off'))=='on':
        choice1 = True
    if request.GET.get("f2", request.session.get('f2', 'off'))=='on':
        choice2 = True

    return StreamingHttpResponse(
        generate_camera_stream(camera_index, apply_detection1=choice1, apply_detection2=choice2 ),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def save_detection(frame, detection_label):
    global last_save_time
    
    # Get the current time
    current_time = time.time()
    
    # Check if 1 second has passed since the last save
    if current_time - last_save_time >= 1:
        # Update the last save time
        last_save_time = current_time
        
        # Convert OpenCV frame to PIL image
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save the image to an in-memory file
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_file = ContentFile(buffer.getvalue(), f"{timezone.now().strftime('%Y%m%d_%H%M%S')}_{detection_label}.jpg")
        
        # Create a new record in the database
        detection_record = DoorDetection(result=detection_label)
        detection_record.image.save(image_file.name, image_file)
        detection_record.save()

def data(request, image_name):
    # Try to retrieve the image data from the database by its name
    try:
        detection = DoorDetection.objects.get(image="door_detections/"+image_name)
    except DoorDetection.DoesNotExist:
        raise Http404(f"No image found with name {image_name}")
    
    # Serve the image binary data as a response
    return HttpResponse(detection.image, content_type='image/jpeg')