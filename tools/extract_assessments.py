import re

# Read both log files
with open('test_outputs_multicam/camera_0_logi_cam_c920e/run.log', 'r', encoding='utf-8') as f:
    log_cam0 = f.read()

with open('test_outputs_multicam/camera_1_hd_pro_webcam_c920/run.log', 'r', encoding='utf-8') as f:
    log_cam1 = f.read()

# Extract all Vision Agent assessments
pattern = r'Evaluating: (.+?)\n-+\nPrompt: (.+?)\n-+\nAgent Assessment:\n(.+?)(?=\n-+|$)'

print("=" * 80)
print("CAMERA 0: Logi Cam C920e")
print("=" * 80)
for match in re.finditer(pattern, log_cam0, re.DOTALL):
    image, prompt, assessment = match.groups()
    print(f"\nImage: {image.strip()}")
    print(f"Assessment:\n{assessment.strip()}\n")
    print("-" * 80)

print("\n" + "=" * 80)
print("CAMERA 1: HD Pro Webcam C920")
print("=" * 80)
for match in re.finditer(pattern, log_cam1, re.DOTALL):
    image, prompt, assessment = match.groups()
    print(f"\nImage: {image.strip()}")
    print(f"Assessment:\n{assessment.strip()}\n")
    print("-" * 80)
