make pip:
	pip install -r requirements.txt

pip-dev:
	pip install mkdocs-material mkdocs-glightbox mkdocs-material[imaging] && export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib

make env:
	python3 -m venv venv && source venv/bin/activate

make test:
	pytest

make up:
	pip install -r requirements.txt
	cd src && python -m deployment.app

make deploy:
	docker-compose -f src/deployment/docker-compose.yaml up --build