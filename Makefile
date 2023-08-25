generate-scenarios:
	python data/generate_topic_scenarios.py

generate-seed-instances:
	python data/generate_seed_instances.py

format:
	black .

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +


