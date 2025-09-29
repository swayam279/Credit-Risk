# 1. Start from a lean, official Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's layer caching
COPY Requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port that Flask runs on. Hugging Face uses port 7860 by default.
EXPOSE 7860

# 7. Define the command to start the Flask application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "app:app"]