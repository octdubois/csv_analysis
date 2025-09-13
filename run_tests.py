#!/usr/bin/env python3
"""
Simple test runner for CSV Insight app.
"""

import sys
import os
import subprocess

def run_tests():
    """Run the test suite."""
    print("🧪 Running CSV Insight Tests...")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
    
    # Run tests
    test_dir = "tests"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory {test_dir} not found")
        return False
    
    print(f"📁 Running tests from {test_dir}")
    
    # Run pytest
    result = pytest.main([
        test_dir,
        "-v",
        "--tb=short"
    ])
    
    if result == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
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
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available!")
    return True

def main():
    """Main test runner."""
    print("🚀 CSV Insight Test Runner")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return 1
    
    print()
    
    # Run tests
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test run completed successfully!")
        return 0
    else:
        print("💥 Test run completed with failures!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 