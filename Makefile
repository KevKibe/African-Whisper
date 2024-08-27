make pip:
	pip install -r requirements.txt

make env:
	python3 -m venv venv && source venv/bin/activate

make test:
	pytest

make up:
	cd src && python -m deployment.app

make deploy:
	docker-compose -f src/deployment/docker-compose.yaml up --build