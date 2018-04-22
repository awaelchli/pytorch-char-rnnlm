FROM ubuntu:16.04

#-------------------------------------------------------------------------------
### Enable UTF8 in docker instance
#-------------------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y locales && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### Stuffs....
#-------------------------------------------------------------------------------

RUN apt-get update -y && \
    apt-get install -y coreutils python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update -y && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app

RUN wget https://srv0.alantian.net/public/share/pytorch-char-rnnlm/model.tar.gz && \
    tar zxvf model.tar.gz

COPY *.py pip_freeze /app/
COPY hps /app/hps/
COPY sampler-hps /app/sampler-hps/

RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision

RUN pip3 install -r pip_freeze

EXPOSE 80
CMD [ "python3", "./sample-http.py", "-f", "sampler-hps/jinyong.json", "-f", "sampler-hps/akb.json", "--host", "0.0.0.0", "-p", "80" ]
