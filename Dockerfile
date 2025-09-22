# In your Dockerfile

# 1. Start from a lean, official Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install the Python dependencies
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port that Flask runs on
EXPOSE 5000

# 7. Define the command to start the Flask application
# The --host=0.0.0.0 is crucial to make the app accessible from outside the container
CMD ["flask", "run", "--host=0.0.0.0"]