kaiba
=====

[![Docker Automated buil](https://img.shields.io/docker/automated/mtpatter/kaiba.svg)](https://hub.docker.com/r/mtpatter/kaiba/)

Bootlier implementation for anomaly detection

Build Docker container for notebooks with dependencies:

```
$ docker build -t "kaiba" .
```

Run Jupyter notebooks like this:

```
$ docker run -it --rm \
	-v $PWD:/home/jovyan/work:rw \
	-p 8888:8888 \
	kaiba
```
