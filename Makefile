# Makefile for setting up and running the web application

# Install required dependencies
install:
	pip install -r requirements.txt

# Run the application on http://localhost:3000
run:
	python app.py