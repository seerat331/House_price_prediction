from pathlib import Path 

def create_directories(*directories):
    for directory in directories:
        Path(directory).mkdir(
            parents=True,
            exist_ok=True
        )

def separator(length=60):
    print("="*length)

def print_heading(title):
    separator()
    print(title)
    separator()