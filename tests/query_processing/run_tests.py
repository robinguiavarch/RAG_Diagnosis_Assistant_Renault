"""
run_tests
=========

Simple utility script to execute all unit tests for the `query_processing` module.

Each test script is run sequentially using pytest, and a summary of the
number of passed and failed files is printed to the console.
"""

import os
import subprocess
import sys


def run_tests() -> None:
    """Execute all unit test scripts in the query_processing module directory."""

    test_dir = os.path.dirname(__file__)

    test_files = [
        "test_llm_client.py",
        "test_response_parser.py",
        "test_unified_processor.py",
        "test_enhanced_retrieval.py",
        "test_integration.py",
    ]

    print("Running unit tests for query_processing")
    print("=" * 50)

    total_tests = 0
    passed_tests = 0
    failed_files = []

    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)

        if not os.path.exists(test_path):
            print(f"[SKIPPED] Missing file: {test_file}")
            continue

        print(f"\n▶ Running: {test_file}")
        print("-" * 30)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True,
            )
            passed_tests += 1
        except subprocess.CalledProcessError:
            print(f"[FAILED] {test_file}")
            failed_files.append(test_file)

        total_tests += 1

    print("\nTest summary")
    print("=" * 50)
    print(f"Total test scripts run: {total_tests}")
    print(f"Passed               : {passed_tests}")
    print(f"Failed               : {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f" - {f}")


if __name__ == "__main__":
    run_tests()
