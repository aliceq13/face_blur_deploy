import os
import sys
import django
from pathlib import Path

# Setup Django environment
sys.path.append(str(Path(__file__).resolve().parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

from apps.videos.models import Video

def check_latest_video():
    try:
        video = Video.objects.last()
        if not video:
            print("No videos found.")
            return

        print(f"Video ID: {video.id}")
        print(f"Title: {video.title}")
        print(f"Status: {video.status}")
        print(f"Progress: {video.progress}")
        print(f"Processed File URL: {video.processed_file_url}")
        
        if video.processed_file_url:
            # Check if file exists in media root
            from django.conf import settings
            # Remove /media/ prefix if present
            rel_path = str(video.processed_file_url).replace(settings.MEDIA_URL, '')
            full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
            print(f"Full Path: {full_path}")
            print(f"File Exists: {os.path.exists(full_path)}")
            
        print("-" * 30)
        print("Processing Jobs:")
        for job in video.processing_jobs.all():
            print(f"Job ID: {job.id}")
            print(f"Type: {job.job_type}")
            print(f"Status: {job.status}")
            print(f"Error: {job.error_message}")
            print(f"Result: {job.result_data}")
            print("-" * 10)
            
        # Check HTTP accessibility
        if video.processed_file_url:
            import requests
            url = f"http://localhost:8000{video.processed_file_url}"
            print(f"Checking URL: {url}")
            try:
                response = requests.head(url)
                print(f"HTTP Status: {response.status_code}")
                print(f"Content-Type: {response.headers.get('Content-Type')}")
                print(f"Content-Length: {response.headers.get('Content-Length')}")
            except Exception as e:
                print(f"HTTP Request Failed: {e}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_latest_video()
