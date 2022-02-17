FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

LABEL maintainer="Manda"

ARG user=KMNER


RUN apt-get update && apt-get install -y --no-install-recommends python3.8 python3-setuptools python3-pip apt-file vim

COPY . /home/$user

WORKDIR /home/$user

# please write your dependencies in requirements.txt
RUN python3.8 -m pip install --upgrade pip && python3.8 -m pip install -r requirements.txt

# set working dir
WORKDIR /home/$user/app

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD ["python3"]
