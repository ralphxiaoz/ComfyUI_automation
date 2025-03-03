import pytest
import os
import sys

def main():
    """Run all tests"""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create necessary test directories if they don't exist
    os.makedirs('tests/mock_data/workflow', exist_ok=True)
    os.makedirs('tests/mock_data/res', exist_ok=True)
    
    # Run pytest with coverage
    exit_code = pytest.main([
        "tests",
        "-v",
        "--cov=code",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 