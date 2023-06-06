.PHONY: start
start:
	uvicorn main:app --host 0.0.0.0 --port 7227

.PHONY: format
format:
	black .
	isort .