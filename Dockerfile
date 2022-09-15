FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y

RUN apt-get install -y python3-pip python-dev build-essential

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Copy all files from current folder (.) to container's folder (.)
COPY . . 

# Set working directory container's default folder (.)
WORKDIR .

RUN pip install -r requirements.txt

# Define which program to run when container starts
ENTRYPOINT [ "python3" ]

# Pass file as parameter to the entry command to start your app
CMD [ "app.py" ]
