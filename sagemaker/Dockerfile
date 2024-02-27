# Start from an official Python base image
FROM python:3.10-slim-buster

# (Optional) Set a working directory to install the dependencies
WORKDIR /opt/program

# Copy the requirements file into the image
COPY requirements.txt /opt/program/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Note: No CMD or specific script is defined here, as this is a base image
