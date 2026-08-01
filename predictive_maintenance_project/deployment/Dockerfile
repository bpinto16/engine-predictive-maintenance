# Minimal base image with Python 3.9-slim installed
FROM python:3.9-slim

# Create a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install deps first
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Then copy the rest of the app
COPY --chown=user . .

# Define the command to run the Streamlit app on port "8501" and it is accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
