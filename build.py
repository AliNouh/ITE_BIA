#!/usr/bin/env python
"""
ITE_BIA Project Build Script
This script performs a complete build of the project when executed
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Execute a command and display a descriptive message"""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ Command executed successfully!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🔍 Checking Python version...")
    if sys.version_info >= (3, 8):
        print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is supported")
        return True
    else:
        print(f"❌ Python version {sys.version_info.major}.{sys.version_info.minor} is not supported. Required: 3.8+")
        return False

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        result = run_command("python -m venv .venv", "Creating virtual environment")
        if not result:
            return False
    else:
        print("✅ Virtual environment already exists")
    return True

def activate_virtual_environment():
    """Activate the virtual environment"""
    if os.name == 'nt':  # Windows
        activate_script = ".venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        activate_script = ".venv/bin/activate"
    
    print(f"🔧 Activating virtual environment: {activate_script}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📚 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements from requirements.txt
    if os.path.exists("requirements.txt"):
        if not run_command("python -m pip install -r requirements.txt", "Installing from requirements.txt"):
            return False
    else:
        print("⚠️ requirements.txt not found, installing core dependencies...")
        core_deps = [
            "numpy>=1.20.0,<2.0.0",
            "pandas>=1.3.0,<2.0.0", 
            "scikit-learn>=1.0.0",
            "streamlit>=1.15.0",
            "plotly>=5.3.0",
            "matplotlib>=3.4.0"
        ]
        
        for dep in core_deps:
            if not run_command(f"python -m pip install {dep}", f"Installing {dep}"):
                return False
    
    return True

def install_project_in_editable_mode():
    """Install project in editable mode"""
    print("🔧 Installing project in editable mode...")
    if os.path.exists("setup.py"):
        if not run_command("python -m pip install -e .", "Installing project in editable mode"):
            return False
    return True

def run_tests():
    """Run tests if they exist"""
    test_dir = Path("tests")
    if test_dir.exists():
        print("🧪 Running tests...")
        if not run_command("python -m pytest tests/ -v", "Running pytest tests"):
            print("⚠️ Some tests failed, but continuing...")
    else:
        print("ℹ️ No tests directory found, skipping tests")

def create_build_artifacts():
    """Create build artifacts"""
    print("📦 Creating build artifacts...")
    
    # Create README.md if it doesn't exist
    if not os.path.exists("README.md"):
        readme_content = """# ITE_BIA Project

An AI research and data processing project for genetic feature selection.

## Requirements

Python 3.8+

## Installation and Building

To automatically build the project, run:

```bash
python build.py
```

## Usage

```bash
streamlit run web/app.py
```

## Structure

- `src/`: Main source code
- `web/`: Web interface using Streamlit
- `tests/`: Unit tests
- `examples/`: Sample data files
"""
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✅ Created README.md")

def verify_build():
    """Verify build success"""
    print("🔍 Verifying build...")
    
    # Check for required core libraries
    required_modules = ['numpy', 'pandas', 'sklearn', 'streamlit', 'plotly']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is installed")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} is missing")
    
    if missing_modules:
        print(f"❌ Missing modules: {', '.join(missing_modules)}")
        return False
    
    print("✅ All required modules are installed")
    return True

def main():
    """Main function for project building"""
    print("🚀 Starting ITE_BIA Project Build")
    print("=" * 60)
    
    # Check version
    if not check_python_version():
        print("❌ Build failed: Python version check failed")
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Build failed: Virtual environment creation failed")
        return 1
    
    # Activate virtual environment
    if not activate_virtual_environment():
        print("❌ Build failed: Virtual environment activation failed")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Build failed: Dependencies installation failed")
        return 1
    
    # Install project
    if not install_project_in_editable_mode():
        print("❌ Build failed: Project installation failed")
        return 1
    
    # Run tests
    run_tests()
    
    # Create build artifacts
    create_build_artifacts()
    
    # Verify build
    if not verify_build():
        print("❌ Build verification failed")
        return 1
    
    print("\n" + "="*60)
    print("🎉 Build completed successfully!")
    print("="*60)
    print("\n🚀 To run the application:")
    print("   streamlit run web/app.py")
    print("\n📚 For development:")
    print("   python -m pytest tests/")
    print("\n🔧 For project management:")
    print("   python setup.py develop")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
