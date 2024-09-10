pip:
	pip install -r requirements.txt

pip-dev:
	pip install mkdocs-material mkdocs-glightbox mkdocs-material[imaging] && export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

env:
	python3 -m venv venv && source venv/bin/activate

test:
	pytest

up:
	cd src/deployment && pip install -r requirements.txt && cd .. && python -m deployment.app

deploy:
	docker-compose -f src/deployment/docker-compose.yaml up --build