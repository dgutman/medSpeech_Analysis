#!/usr/bin/env python3
"""
Test script to verify the setup of the Medical Speech Analysis Results Browser.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def test_environment():
    """Test environment setup"""
    print("🔍 Testing environment setup...")
    
    # Check if .env file exists
    env_file = Path("../.env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for required variables
        with open(env_file, 'r') as f:
            content = f.read()
            if 'PIXELTABLE_API_KEY' in content:
                print("✅ PIXELTABLE_API_KEY found in .env")
            else:
                print("⚠️  PIXELTABLE_API_KEY not found in .env")
    else:
        print("⚠️  .env file not found - using sample data")
    
    # Check directories
    cache_dir = Path("./cache")
    data_dir = Path("./data")
    
    cache_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"✅ Cache directory: {cache_dir.absolute()}")
    print(f"✅ Data directory: {data_dir.absolute()}")
    
    return True

def test_dependencies():
    """Test Python dependencies"""
    print("\n🔍 Testing Python dependencies...")
    
    required_packages = [
        'dash', 'pandas', 'numpy', 'plotly', 
        'dash_bootstrap_components', 'dash_ag_grid'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_data_preloading():
    """Test data preloading"""
    print("\n🔍 Testing data preloading...")
    
    try:
        # Import and run preload_data
        from preload_data import main
        df = main()
        
        if df is not None and len(df) > 0:
            print(f"✅ Data loaded successfully: {len(df)} records")
            print(f"✅ Columns: {list(df.columns)}")
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Data preloading failed: {e}")
        return False

def test_docker_setup():
    """Test Docker setup"""
    print("\n🔍 Testing Docker setup...")
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found")
        return False
    
    # Check if docker-compose is available
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose not found")
        return False
    
    # Check if Dockerfile exists
    if Path("Dockerfile").exists():
        print("✅ Dockerfile found")
    else:
        print("❌ Dockerfile not found")
        return False
    
    # Check if docker-compose.yml exists
    if Path("docker-compose.yml").exists():
        print("✅ docker-compose.yml found")
    else:
        print("❌ docker-compose.yml not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Medical Speech Analysis Results Browser - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment),
        ("Python Dependencies", test_dependencies),
        ("Data Preloading", test_data_preloading),
        ("Docker Setup", test_docker_setup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! You're ready to run the application.")
        print("\nTo start the application:")
        print("  docker-compose up --build")
        print("\nOr manually:")
        print("  python app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


