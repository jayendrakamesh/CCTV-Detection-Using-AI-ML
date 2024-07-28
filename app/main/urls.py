from django.urls import path
from . import views
from django.http import HttpResponse

urlpatterns = [
    path('', views.user_login, name='login'),
    path('home/', views.home, name='home'),
    path('movement_feed/', views.movement_feed, name='movement_feed'),
    path('ppe_feed/', views.ppe_feed, name='ppe_feed'),
    path('door_feed/', views.door_feed, name='door_feed'),
]
