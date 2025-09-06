#!/usr/bin/env python3
"""
Quick start script for CSV Insight app.
"""

import os
import sys
import subprocess
import time

def check_ollama():
    """Check if Ollama is running."""
    print("🔍 Checking Ollama status...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            return True
        else:
            print("❌ Ollama responded but with error status")
            return False
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        return False

def check_dependencies():
    """Check if Python dependencies are installed."""
    print("🔍 Checking Python dependencies...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "pydantic",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available!")
    return True

def install_dependencies():
    """Install missing dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_ollama_model():
    """Download a default Ollama model."""
    print("🤖 Downloading Ollama model...")
    
    try:
        # Try to download llama3.1:8b
        subprocess.run([
            "ollama", "pull", "llama3.1:8b"
        ], check=True)
        print("✅ Model downloaded successfully!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Failed to download model: {e}")
        print("💡 Make sure Ollama is installed and 'ollama' command is available")
        return False

def start_app():
    """Start the Streamlit app."""
    print("🚀 Starting CSV Insight app...")
    
    try:
        # Check if app.py exists
        if not os.path.exists("app.py"):
            print("❌ app.py not found in current directory")
            return False
        
        print("🌐 App will open in your browser at http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the app")
        print("\n" + "=" * 50)
        
        # Start the app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        return False

def main():
    """Main quick start function."""
    print("🚀 CSV Insight - Quick Start")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please run manually:")
            print("   pip install -r requirements.txt")
            return 1
        print()
    
    # Step 2: Check Ollama
    if not check_ollama():
        print("\n⚠️  Ollama is not running. Please:")
        print("   1. Install Ollama from https://ollama.ai/")
        print("   2. Start Ollama: ollama serve")
        print("   3. Download a model: ollama pull llama3.1:8b")
        print("\n💡 After starting Ollama, run this script again.")
        return 1
    
    # Step 3: Check for models
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get("models", [])
        
        if not models:
            print("\n🤖 No models found. Downloading default model...")
            if not download_ollama_model():
                print("❌ Failed to download model. Please run manually:")
                print("   ollama pull llama3.1:8b")
                return 1
        else:
            print(f"✅ Found {len(models)} model(s): {[m['name'] for m in models]}")
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return 1
    
    # Step 4: Start the app
    print("\n🎯 All checks passed! Starting the app...")
    time.sleep(2)
    
    return start_app()

if __name__ == "__main__":
    sys.exit(main()) 