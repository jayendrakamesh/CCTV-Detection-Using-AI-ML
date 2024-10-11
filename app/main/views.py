from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import cv2 as cv
import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Load models with error handling and map_location for CPU/GPU compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    movement_model = torch.load('main/movement.pth', map_location=device)
    door_model = torch.load('main/door.pth', map_location=device)
    ppe_model = torch.load('main/PPE.pth', map_location=device)
except Exception as e:
    print(f"Error loading models: {e}")

# User login view
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

# Function to transform frame for model
def transform_frame_for_model(frame):
    frame_resized = cv.resize(frame, (640, 640))  # Resize to model input size
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  # Convert to tensor and normalize
    return frame_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to the appropriate device

# Function to overlay result on frame
def overlay_result_on_frame(frame, result):
    # Assuming result contains bounding boxes and classes
    for box, cls in zip(result['boxes'], result['classes']):
        x1, y1, x2, y2 = box  # Assuming box is a list of [x1, y1, x2, y2]
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(frame, str(cls), (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Function to apply AI model on frame
def apply_model_on_frame(frame, model):
    model.eval()  # Set model to evaluation mode
    transformed_frame = transform_frame_for_model(frame)  # Transform frame
    with torch.no_grad():  # Disable gradient calculation
        result = model(transformed_frame)  # Run model
    processed_frame_with_result = overlay_result_on_frame(frame, result)  # Overlay results
    return processed_frame_with_result

# Home view to render camera selection
def home(request):
    if request.method == 'POST':
        apply_movement = request.POST.get('f1') == 'on'
        apply_door = request.POST.get('f2') == 'on'
        apply_ppe = request.POST.get('f3') == 'on'

        # Store the choices in session
        request.session['apply_movement'] = apply_movement
        request.session['apply_door'] = apply_door
        request.session['apply_ppe'] = apply_ppe

        return redirect('home')

    # Detect connected cameras
    cameras = []
    for i in range(3):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cameras.append({'index': i, 'resolution': (width, height)})
        cap.release()

    context = {'cameras': cameras}
    return render(request, 'main/home.html', context)

# Function to resize video with black bars to maintain aspect ratio
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

# Generator function to stream frames with AI detection based on user choices
def generate_camera_stream(camera_index, apply_movement, apply_door, apply_ppe):
    try:
        cap = cv.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Camera {camera_index} cannot be opened.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = resize_with_black_bars(frame)

            # Apply models based on user choices
            if apply_movement:
                frame = apply_model_on_frame(frame, movement_model)
            if apply_door:
                frame = apply_model_on_frame(frame, door_model)
            if apply_ppe:
                frame = apply_model_on_frame(frame, ppe_model)

            # Encode the frame as JPEG
            ret, jpeg = cv.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    except Exception as e:
        print(f"Error with camera {camera_index}: {e}")
    finally:
        cap.release()

# Stream camera view
def stream_camera(request, camera_index):
    # Retrieve model application choices from session
    apply_movement = request.session.get('apply_movement', False)
    apply_door = request.session.get('apply_door', False)
    apply_ppe = request.session.get('apply_ppe', False)

    return StreamingHttpResponse(
        generate_camera_stream(camera_index, apply_movement, apply_door, apply_ppe),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
