"""Smoke tests for the project package."""


def test_app_package_imports() -> None:
    """The application package should be importable."""
    import app

    assert app.__version__ == "0.1.0"

