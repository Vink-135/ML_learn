# Base image
FROM python:3.9

# Workdir
WORKDIR /app

# Copy
COPY . /app

# Install Flask
RUN pip install flask

# Port
EXPOSE 8000

# Command
CMD [ "python", "app.py" ]



