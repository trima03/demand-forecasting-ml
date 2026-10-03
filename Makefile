.PHONY: install train serve

install:
	python -m pip install -r requirements.txt

train:
	python -m forecasting.train --output-dir artifacts

serve:
	uvicorn forecasting.api:app --host 0.0.0.0 --port 8000
