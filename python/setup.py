import os
import sys
import subprocess
import venv
import platform
from pathlib import Path

def get_app_path():
    """Get the path to the application's resources directory."""
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        if platform.system() == 'Darwin':  # macOS
            return Path(os.path.dirname(sys.executable)).parent / 'Resources'
        elif platform.system() == 'Windows':
            return Path(os.path.dirname(sys.executable))
        else:  # Linux
            return Path(os.path.dirname(sys.executable))
    else:
        # Running in normal Python environment
        return Path(__file__).parent

def create_venv(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully!")

def get_python_executable(venv_path):
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'python.exe'
    return venv_path / 'bin' / 'python'

def install_requirements(venv_path):
    """Install required packages in the virtual environment."""
    python_exe = get_python_executable(venv_path)
    # Look for requirements.txt in the same directory as setup.py
    requirements_path = get_app_path() / 'requirements.txt'
    
    if not requirements_path.exists():
        print("Error: requirements.txt not found!")
        print(f"Looking for requirements.txt at: {requirements_path}")
        return False

    print("Installing required packages...")
    try:
        subprocess.run([
            str(python_exe),
            '-m', 'pip', 'install',
            '--upgrade', 'pip'
        ], check=True)
        
        subprocess.run([
            str(python_exe),
            '-m', 'pip', 'install',
            '-r', str(requirements_path)
        ], check=True)
        
        print("Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def setup():
    """Main setup function."""
    app_path = get_app_path()
    venv_path = app_path / 'python' / 'venv'
    
    print(f"App path: {app_path}")
    print(f"Venv path: {venv_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Is frozen: {getattr(sys, 'frozen', False)}")
    
    # Create virtual environment
    create_venv(venv_path)
    
    # Install requirements
    if install_requirements(venv_path):
        # Create a marker file to indicate successful setup
        (venv_path / '.setup_complete').touch()
        return True
    return False

if __name__ == '__main__':
    success = setup()
    sys.exit(0 if success else 1) 