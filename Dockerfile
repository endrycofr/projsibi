# Use a more up-to-date Python base image
FROM python:3.11-slim-buster

# Expose the port that Streamlit will run on
EXPOSE 8501

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    software-properties-common \
    git \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip, setuptools, and wheel to avoid potential issues
RUN pip3 install --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with increased verbosity for debugging
RUN pip3 install -r requirements.txt --verbose


# Copy the rest of the application code
COPY . .

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
