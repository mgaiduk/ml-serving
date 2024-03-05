# Start from an official Python base image
FROM python:3.10-slim-buster

# (Optional) Set a working directory to install the dependencies
WORKDIR /opt/program

# Copy the requirements file into the image
COPY code/requirements.txt /opt/program/
RUN mkdir code
COPY code/*.py code
# copy requirements to the code folder to make it available to nested sub scripts as well
COPY code/requirements.txt code

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Note: No CMD or specific script is defined here, as this is a base image
