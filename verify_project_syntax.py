import py_compile
import sys
import os

files_to_check = [
    '/app/apps/videos/tasks.py',
    '/app/CVLface/cvlface/general_utils/huggingface_model_utils.py',
    '/app/apps/videos/adaface_wrapper.py',
    '/app/apps/videos/face_aligner.py',
    '/app/apps/videos/face_instance_detector.py',
]

print("Starting Syntax Check...")
has_error = False

for file_path in files_to_check:
    try:
        if not os.path.exists(file_path):
             print(f"[MISSING] {file_path}")
             has_error = True
             continue
             
        py_compile.compile(file_path, doraise=True)
        print(f"[OK] {file_path}")
    except py_compile.PyCompileError as e:
        print(f"[ERROR] {file_path}: {e}")
        has_error = True
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        has_error = True

if has_error:
    print("\nCheck FAILED with errors.")
    sys.exit(1)
else:
    print("\nAll files passed syntax check.")
    sys.exit(0)
