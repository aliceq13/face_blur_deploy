import os
import sys
import django
from pathlib import Path

# Setup Django environment
sys.path.append(str(Path(__file__).resolve().parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

from apps.videos.models import Video
from apps.videos.tasks import process_video_blur_task

def trigger_task():
    try:
        video = Video.objects.last()
        if not video:
            print("No videos found.")
            return

        print(f"Triggering task for Video ID: {video.id}")
        
        # Trigger task
        task = process_video_blur_task.delay(str(video.id))
        print(f"Task triggered. Task ID: {task.id}")
        
        # Update status manually to 'processing' to mimic view logic (optional but good for consistency)
        # But the task itself should update it. Let's see if the task runs.
        
    except Exception as e:
        print(f"Error triggering task: {e}")

if __name__ == "__main__":
    trigger_task()
