ARG BASE_IMAGE=sinzlab/pytorch:v3.8-torch1.5.0-cuda10.2-dj0.12.7
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
RUN git clone https://github.com/andererka/deepgaze_pytorch.git &&\
    git clone https://github.com/andererka/nnvision.git
    git clone https://github.com/sinzlab/nnfabrik &&\

FROM ${BASE_IMAGE}
COPY --from=base /src /src
ADD . /src/nnsaliency


RUN pip install -e ~/lib/deepgaze_pytorch &&\
    pip install -e ~/lib/nnvision &&\

