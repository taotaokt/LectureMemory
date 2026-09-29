"""Project-root Streamlit launcher.

Keeping the launcher at the repository root ensures the application packages are
importable when Streamlit executes the file directly.
"""

from frontend.streamlit_app import main

if __name__ == "__main__":
    main()
