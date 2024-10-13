from django.contrib import admin
from .models import DoorDetection

@admin.register(DoorDetection)
class DoorDetectionAdmin(admin.ModelAdmin):
    list_display = ('detection_time', 'result', 'image')  # Show these fields in the list view
