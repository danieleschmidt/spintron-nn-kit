#!/usr/bin/env python3
"""Deployment validation script."""

import sys
import subprocess
import json
from pathlib import Path

def validate_installation():
    """Validate SpinTron-NN-Kit installation."""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import spintron_nn; print('Installation OK')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Installation validation passed")
            return True
        else:
            print(f"❌ Installation validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation validation error: {e}")
        return False

def validate_configuration():
    """Validate configuration files."""
    config_files = ["config.json", "production.json"]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file) as f:
                    json.load(f)
                print(f"✅ Configuration {config_file} is valid")
            except Exception as e:
                print(f"❌ Configuration {config_file} error: {e}")
                return False
        else:
            print(f"⚠️  Configuration {config_file} not found")
    
    return True

def main():
    """Run deployment validation."""
    print("SpinTron-NN-Kit Deployment Validation")
    print("=" * 40)
    
    checks = [
        ("Installation", validate_installation),
        ("Configuration", validate_configuration)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\nRunning {name} validation...")
        if not check_func():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All deployment validations passed!")
        return 0
    else:
        print("\n💥 Some deployment validations failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
