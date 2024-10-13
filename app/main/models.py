from django.db import models

class DoorDetection(models.Model):
    detection_time = models.DateTimeField(auto_now_add=True)  # Automatically set the time when the record is created
    image = models.ImageField(upload_to='door_detections/')  # Path to store the detection image
    result = models.CharField(max_length=255)  # Store the result of detection (e.g., "Open Door", "Closed Door")

    def __str__(self):
        return f"Detected {self.result} at {self.detection_time}"
