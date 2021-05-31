PYTHON = python3
PIP = pip3
GIT = git

install:
	 $(PIP) install -r requirements.txt

main:
	$(PYTHON) main.py