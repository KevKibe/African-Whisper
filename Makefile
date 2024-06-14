make pip:
	pip install -r requirements.txt

make env:
	python3 -m venv venv && source venv/bin/activate
