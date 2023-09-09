"""Miscellaneous utilities"""
from typing import List, Any
import json


def read_text_file(file_path: str) -> str:
    """Helper to read a text file into a string"""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return content


def write_text_file(file_path: str, text: str) -> str:
    """Helper to write a string into a text file"""
    with open(file_path, "w") as file:
        file.write(text)


def append_to_file(filename, text):
    """Helper to iteratively append text to a file"""
    with open(filename, "a") as f:
        f.write(text)
        f.flush()


def read_text_file_lines(file_path: str) -> List[str]:
    "Helper to read a text file line by line into a list"
    res = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            res.append(line.strip())

    return res


def write_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(obj, json_file, indent=4)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as json_file:
        obj = json.load(json_file)

    return obj
