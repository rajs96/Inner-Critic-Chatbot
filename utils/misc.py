"""Miscellaneous utilities"""

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()

    return content