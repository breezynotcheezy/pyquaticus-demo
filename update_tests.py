import os
import re
from pathlib import Path

def update_test_file(file_path):
    """Update a test file to work with the current API."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Common replacements
        replacements = [
            (r'from pyquaticus import pyquaticus_v0', 
             'from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team'),
            (r'env = pyquaticus_v0\.PyQuaticusEnv\(', 'env = PyQuaticusEnv('),
            (r'obs, reward, term, trunc, info = env\.step\(', 'next_obs_dict, rewards, dones, infos = env.step('),
            (r'if term\[k\[0\]\] == True or trunc\[k\[0\]\]==True:', 'if any(dones.values()):'),
            (r'env\.reset\(\)', 'env.reset()'),
            (r'env\.close\(\)', 'env.close()')
        ]
        
        for old, new in replacements:
            content = re.sub(old, new, content)
            
        # Add imports if needed
        if 'import numpy as np' not in content:
            content = 'import numpy as np\n' + content
            
        # Add error handling if not present
        if 'try:' not in content and 'except KeyboardInterrupt:' not in content:
            # Find the main loop
            main_loop = re.search(r'while.*?:', content)
            if main_loop:
                start = main_loop.start()
                content = content[:start] + 'try:\n' + content[start:]
                content += '\nexcept KeyboardInterrupt:\n    print("Test stopped by user")\nfinally:\n    env.close()\n    print("Environment closed.")'
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Updated {file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating {file_path}: {str(e)}")
        return False

def main():
    test_dir = Path(__file__).parent / 'test'
    updated = 0
    
    for file_path in test_dir.glob('*.py'):
        if file_path.name != 'update_tests.py':  # Skip this script
            if update_test_file(file_path):
                updated += 1
    
    print(f"\nUpdated {updated} test files.")

if __name__ == '__main__':
    main()
