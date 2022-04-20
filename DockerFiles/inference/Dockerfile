FROM pytorch/pytorch:latest

MAINTAINER Siddhartha Singh "ss13793@nyu.edu"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --assume-yes git

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /app && cd /app \
 && git clone --progress --verbose https://github.com/Black-Caesaris/mnist_kubernetes.git

WORKDIR /app/mnist_kubernetes

RUN /opt/conda/bin/conda install --yes pandas  
RUN /opt/conda/bin/conda install --yes numpy  
RUN /opt/conda/bin/conda install --yes scikit-learn

ENV PORT 8001
EXPOSE $PORT

CMD ["python", "inference.py", "--epochs", "1"]