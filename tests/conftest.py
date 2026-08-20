"""
Shared pytest configuration for the physys integration test.
"""


def pytest_addoption(parser):

    parser.addoption(
        "--physys-config",

        action = "store",
        default = None,
        help = "Path to config.json to run the physys integration test against.",
    )
