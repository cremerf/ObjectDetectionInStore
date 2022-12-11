FROM python:3.9 as base

# Update pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install aiofiles
RUN pip install python-multipart

# Set python path
ENV PYTHONPATH=$PYTHONPATH:/home

# Set workdir
WORKDIR /home

# Copy info inside 
ADD . /home/api

# FROM base as test
# RUN ["pytest", "-v", "/src/tests"]

# Start the api service
FROM base as build
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]