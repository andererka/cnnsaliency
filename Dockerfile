ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials
RUN git clone -b attention_readout https://github.com/KonstantinWilleke/neuralpredictors &&\
    git clone https://github.com/sinzlab/nnfabrik &&\
    git clone -b konsti_monkey_experiments https://github.com/sinzlab/mei &&\
    git clone https://github.com/sinzlab/data_port &&\
    git clone https://github.com/sinzlab/nexport &&\
    git clone https://github.com/matthias-k/pysaliency

FROM ${BASE_IMAGE}
COPY --from=base /src /src
ADD . /src/nnsaliency

RUN pip install -e /src/neuralpredictors &&\
    pip install -e /src/nnfabrik &&\
    pip install -e /src/mei &&\
    pip install -e /src/data_port &&\
    pip install -e /src/nexport &&\
    pip install -e /src/nnsaliency &&\
    pip install -e /src/nnsaliency/lib/nnvision &&\
    pip install -e /src/nnsaliency/lib/deepgaze_pytorch &&\
    pip install git+https://github.com/dicarlolab/CORnet &&\
    pip install glom &&\
    pip install Cython &&\
    pip3 install git+https://github.com/sacadena/ptrnets &&\
    pip install -e /src/pysaliency &&\
    pip install -U Jinja2

