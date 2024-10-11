from django.urls import path
from . import views
from django.http import HttpResponse

urlpatterns = [
    path('', views.user_login, name='login'),
    path('home/', views.home, name='home'),
    path('stream/<int:camera_index>/', views.stream_camera, name='stream_camera'),
]
