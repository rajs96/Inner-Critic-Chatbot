generate-scenarios:
	python data/generate_topic_scenarios.py

generate-seed-instances:
	python data/generate_seed_instances.py

format:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

