.PHONY: all install help

# Default Python interpreter
PYTHON = python
VENV = .venv/Scripts/activate

# Set PYTHONPATH to include src and utils directories
PYTHONPATH = $(shell cd)\\src;$(shell cd)\\utils

# Default target
all: help

# Help target
help:
	@echo Available targets:
	@echo   make install             - Install project dependencies and set up environment
	@echo   make clean               - Clean up artifacts

# Install project dependencies and set up environment
install:
	@echo Installing project dependencies and setting up environment...
	@echo Creating virtual environment...
	$(PYTHON) -m venv .venv
	@echo Activating virtual environment and installing dependencies...
	.venv\Scripts\activate && python.exe -m pip install --upgrade pip
	.venv\Scripts\activate && pip install -r requirements.txt
	@echo Installation completed successfully!
	@echo To activate the virtual environment, run: .venv\Scripts\activate


