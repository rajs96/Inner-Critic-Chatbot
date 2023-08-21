"""Miscellaneous utilities"""
from typing import List
import asyncio
from functools import partial


async def async_run(func, **kwargs):
    loop = asyncio.get_running_loop()
    func_partial = partial(func, **kwargs)
    return await loop.run_in_executor(None, func_partial)


def read_text_file(file_path: str) -> str:
    """Helper to read a text file into a string"""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    return content


def read_text_file_lines(file_path: str) -> List[str]:
    "Helper to read a text file line by line into a list"
    res = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.strip())

    return res
