.PHONY: install fetch-data features train simulate run_all dashboard test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

fetch-data:
	python src/fetch_data.py

features:
	python src/features.py

train:
	python src/models.py

simulate:
	python src/backtest.py

run_all:
	python main.py run_all

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find data/raw/ -type f ! -name ".gitkeep" -delete
	find data/processed/ -type f ! -name ".gitkeep" -delete
	find models/ -type f -name "*.pkl" -delete
	find reports/ -type f ! -name ".gitkeep" -delete
