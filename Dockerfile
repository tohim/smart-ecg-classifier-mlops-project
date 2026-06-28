# DOCKERFILE - ECG Heartbeat Analysis API

# to run the dockerfile:
    # bash: 
        # docker build -t ecg-api:v1 .
# "-t ecg-api:v1" -> "tag" gives image the name ecg-app and the tag v1
# "." -> "build context" -> current folder (project root) is given over to docker, all but the .dockerignore entries

# FROM: every Dockerfile starts with a base image
# python:3.11-slim is an official pyhton image based on Debian Linux - with only minimal packages needed to run Python.
# "slim" means: no build tools, no documentation, no extras -> keeps final image size small (important for faster up/downloads, less attack surface, lower storage cost)

FROM python:3.11-slim

# LABEL: optional metadata. Doesnt affect functionality, but shows in "docker inspect" and is good documentation practice in real teams
LABEL maintainer="Thomas"
LABEL description="ECG Heartbeat Analysis API - CNN classification + Autoencoder anomaly detection"

# WORKDIR: sets the working directory for ALL subsequent RUN, COPY, CMD instructions. If it doesnt exist, Docker creates it. Usine "/app" is a common convention
WORKDIR /app    

# COPY the requirements and install dependencies first
# needs to be before rest of code: bc of docker caching each layer and requirements and dependencies dont change
COPY requirements.txt .

# RUN: executing a shell command during the IMAGE BUILD (not at container startup - that is what CMD is for)
# pip install --no-cache-dir: prevents pip from storing downloaded packages in a local cache inside the image - not needed as it will never run pip again inside container
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# COPY the actual application code and model files
# everything in .dockerignore is exluded, copies src/api/, src/training/models.py, and models/
COPY src/ src/
COPY models/ models/

# PYTHONPATH tells python which directories to search when importing modules. Setting to "/app" -> look for packages from /app - so "src.training.models" resolves
# correctly to "/app/src/training/models.py" - Regardless of which directory uvicorn was started from inside the container
ENV PYTHONPATH=/app

# EXPOSE: documents that the container listens on port 8000 - but this does not actually publish the port (thats done with -p at "docker run" time) 
# its just documentation/ convention - telling anyone reading the Dockerfile "this services uses port 8000"
EXPOSE 8000

# CMD: command that runs when a container STARTS from this image
# using JSON array format (["cmd", "arg1", ...]) is recommended over shell string format ("cmd arg1") - the JSON form runs the command directly (no shell wrapper)
# means: signals like Ctrl+C are handled correctly by uvicorn
# "--host 0.0.0.0" -> by default uvicorn listens only on localhost (127.0.0.1), which is unreachable from outside the container.
# 0.0.0.0 means -> "listen on ALL network interfaces" - necessary so that traffic coming in via Docker's port mapping (-p 8000:8000) actually reaches uvicorn
# here no "--reload" -> wasteful and potentially unstable within a container (as it would restart automatically when files change) 
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
