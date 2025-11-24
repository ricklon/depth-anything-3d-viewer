import re

# Read both camera logs
logs = {
    'Camera 0 (Logi Cam C920e)': 'test_outputs_multicam/camera_0_logi_cam_c920e/run.log',
    'Camera 1 (HD Pro Webcam C920)': 'test_outputs_multicam/camera_1_hd_pro_webcam_c920/run.log'
}

for camera_name, log_path in logs.items():
    print(f"\n{'='*80}")
    print(f"{camera_name}")
    print('='*80)
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all evaluations
    evaluations = re.findall(r'--- Evaluating (.+?) ---.*?Agent Assessment:\s*(.+?)(?=---|\Z)', content, re.DOTALL)
    
    for filename, assessment in evaluations:
        print(f"\n📊 {filename.strip()}")
        print("-" * 80)
        
        # Extract key phrases
        lines = assessment.strip().split('\n')
        for line in lines[:20]:  # First 20 lines
            line = line.strip()
            if line and not line.startswith('Prompt:'):
                print(f"  {line}")
        
        # Look for ratings
        ratings = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*10', assessment)
        if ratings:
            print(f"\n  ⭐ RATING: {ratings[0]}/10")
        
        print()
