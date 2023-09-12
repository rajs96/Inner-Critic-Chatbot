format:
	black .

clean:
	find . -type d \( -name ".pytest_cache" -o -name "__pycache__" -o -name ".mypy_cache" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +