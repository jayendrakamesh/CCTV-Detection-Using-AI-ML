from django.urls import path
from . import views
from django.http import HttpResponse

urlpatterns = [
    path('', views.user_login, name='login'),
    path('home/', views.home, name='home'),
    path('door_detections/<str:image_name>/', views.data, name='data'),
    path('stream/<int:camera_index>/', views.stream_camera, name='stream_camera'),
]
