load-synthetic-data: generate-scenarios generate-seed-instances

generate-scenarios:
	python data/synthetic/generate_topic_scenarios.py

generate-seed-instances:
	python data/synthetic/generate_seed_instances.py

load-reddit-data: scrape-reddit-data filter-reddit-data sample-reddit-data

scrape-reddit-data:
	python data/reddit/scrape_reddit_data.py

filter-reddit-data:
	python data/reddit/filter_reddit_data.py

sample-reddit-data:
	python data/reddit/sample_reddit_data.py

generate-instructions:
	python data/training_data/generate_instruction_finetuning_data.py

format:
	black .

clean:
	find . -type d \( -name ".pytest_cache" -o -name "__pycache__" -o -name ".mypy_cache" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +


