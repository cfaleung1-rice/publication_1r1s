import sys
import subprocess
from pathlib import Path

def main():
    parent = Path(__file__).resolve().parent
    subdirs = sorted(d for d in parent.iterdir() if d.is_dir() and d.name != '__pycache__')

    for subdir in subdirs:
        script = next(subdir.glob('*.py'))
        result = subprocess.run([sys.executable, str(script)], cwd=str(subdir), capture_output=True, text=True)

        if result.returncode == 0:
            print(f'{script.name} complete')
            if result.stdout.strip():
                print(result.stdout, end='')
        else:
            print(f'{script.name} FAILED')
            print(result.stderr)

if __name__ == '__main__':
    main()