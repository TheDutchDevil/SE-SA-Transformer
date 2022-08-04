FROM continuumio/anaconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .

RUN conda env create -f environment.yml

RUN echo "conda activate sesa" >> ~/.profile

# Activate the environment, and make sure it's activated:
SHELL ["/bin/bash", "--login"]

COPY . .

ENTRYPOINT ["/bin/bash", "--login"]