generate-scenarios:
	python data/generate_topic_scenarios.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

