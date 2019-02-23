# Main Image

FROM frolvlad/alpine-python3

# I have extended the previous package with the necessary packages for development
MAINTAINER Andres Mendez <kajuna0kajuna@gmail.com>


# Environment variables
ENV LANG=C.UTF-8
# Version of matplotlib
ARG MATPLOTLIB_VERSION=3.0.2

# Build dependencies
RUN apk add --no-cache --virtual=.build-dependencies g++ gfortran file binutils musl-dev python3-dev openblas-dev && \
    apk add libstdc++ openblas && \
    apk add --update --no-cache build-base libstdc++ libpng libpng-dev freetype freetype-dev && \
    # Make Python3 as default
    ln -fs /usr/include/locale.h /usr/include/xlocale.h && \
    ln -fs /usr/bin/python3 /usr/local/bin/python && \
    ln -fs /usr/bin/pip3 /usr/local/bin/pip && \
    # Install Python dependencies
    pip3 install -v --no-cache-dir matplotlib==$MATPLOTLIB_VERSION && \
    pip3 install -v --no-cache-dir --upgrade pip && \
    # Install the Basic packages
    pip install numpy && \
    pip install pandas && \
    pip install scipy && \
    pip install scikit-learn && \
    # Cleaning all dependencies
    rm -r /root/.cache && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -r '{}' + && \
    find /usr/lib/python3.*/site-packages/ -name '*.so' -print -exec sh -c 'file "{}" | grep -q "not stripped" && strip -s "{}"' \; && \
    rm /usr/include/xlocale.h && \
    rm -vrf /var/cache/apk/*
 
RUN apk add sqlite \
    && apk add bash \
    && apk add vim  \
    && apk add git

RUN pip install --no-cache-dir cython  

# Expose a port for future access to the database
EXPOSE 8080

# Copy Repo files
COPY ./ /Cython_Code/

# Move to the correct path
WORKDIR /Cython_Code


# Entrypoint
ENTRYPOINT /bin/bash
