# Use an official Python runtime as a parent image
#FROM python:3.6
FROM alpine:edge
RUN apk add --repository http://dl-3.alpinelinux.org/alpine/edge/testing/ --no-cache python3 python3-dev py3-pip build-base libffi-dev libgit2-dev bash py3-requests py3-bottle py3-numpy py3-dateutil py3-pygit2 py3-mako py3-scipy

# Set the working directory to /app
WORKDIR /app

ADD . /app

RUN pip3 install --no-cache-dir protobuf

# Run app.py when the container launches
CMD ["./unit_tests/main.py"]
