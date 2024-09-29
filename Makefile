# Makefile for setting up and running the Flask web application

# Define the virtual environment directory
VENV_DIR := venv

# Create and activate the virtual environment, install required dependencies
install:
	python3 -m venv $(VENV_DIR)
	./$(VENV_DIR)/bin/pip install --upgrade pip
	./$(VENV_DIR)/bin/pip install Flask  # Ensure Flask is installed
	./$(VENV_DIR)/bin/pip install -r requirements.txt

# Run the Flask application inside the virtual environment
run:
	FLASK_APP=app FLASK_ENV=development ./$(VENV_DIR)/bin/python -m flask run --host=0.0.0.0 --port=3000

# Clean up: remove the virtual environment after running the application
clean:
	rm -rf $(VENV_DIR)