"""Miscellaneous utilities"""

def read_text_file(file_path: str) -> str:
    """Helper to read a text file into a string"""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return content