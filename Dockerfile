FROM nvidia/cuda
MAINTAINER cutrain <duanyuge@qq.com>
RUN apt-get update
RUN apt-get install -y python3-dev python3-pip
ADD . /competition
COPY ./pip.conf /root/.pip/pip.conf
WORKDIR /competition
RUN pip3 install -r requirements.txt
#Dockerfile of Example
# Version 1.0

# Base Images
#FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu

#MAINTAINER
#MAINTAINER AlibabaSec

#ADD . /competition

#WORKDIR /competition
#RUN pip --no-cache-dir install  -r requirements.txt
# INSTALL cleverhans foolbox

#RUN mkdir ./models
#RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/inception_v1.tar.gz' && tar -xvf inception_v1.tar.gz -C ./models/
