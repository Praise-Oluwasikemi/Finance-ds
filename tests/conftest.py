def pytest_addoption(parser):
    parser.addoption(
        "--practice",
        action="store_true",
        help="Check exercises/practice.py instead of exercises/solutions.py",
    )
