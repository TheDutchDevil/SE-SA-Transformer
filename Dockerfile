FROM gpuci/miniconda-cuda:11.3-base-ubuntu20.04

WORKDIR /app

# Create the environment:
COPY environment.yml .

RUN conda env create -f environment.yml

RUN echo "conda activate sesa" >> ~/.profile

# Activate the environment, and make sure it's activated:
SHELL ["/bin/bash", "--login", "-c"]

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 ignite -c pytorch -c conda-forge -y

COPY . .

ENTRYPOINT ["/bin/bash", "--login"]