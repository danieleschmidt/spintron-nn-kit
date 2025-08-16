#!/usr/bin/env python3
"""Health check script for SpinTron-NN-Kit."""

import sys
import json
import time
from pathlib import Path

def check_imports():
    """Check if core modules can be imported."""
    try:
        import spintron_nn
        return True, "Core modules imported successfully"
    except ImportError as e:
        return False, f"Import error: {e}"

def check_file_permissions():
    """Check file system permissions."""
    try:
        test_file = Path("health_check.tmp")
        test_file.write_text("test")
        test_file.unlink()
        return True, "File system permissions OK"
    except Exception as e:
        return False, f"File permission error: {e}"

def check_memory():
    """Check available memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 1.0:
            return False, f"Low memory: {available_gb:.1f}GB available"
        return True, f"Memory OK: {available_gb:.1f}GB available"
    except ImportError:
        return True, "Memory check skipped (psutil not available)"

def main():
    """Run all health checks."""
    checks = [
        ("imports", check_imports),
        ("file_permissions", check_file_permissions),
        ("memory", check_memory)
    ]
    
    results = {}
    all_passed = True
    
    for name, check_func in checks:
        passed, message = check_func()
        results[name] = {"passed": passed, "message": message}
        if not passed:
            all_passed = False
            
    # Output results
    health_status = {
        "timestamp": time.time(),
        "overall_status": "healthy" if all_passed else "unhealthy",
        "checks": results
    }
    
    print(json.dumps(health_status, indent=2))
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
