FROM condaforge/mambaforge

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV TZ=Europe/Berlin
ENV TERM xterm-256color
ARG PYTHON_VERSION=3.6
ENV PYTHON_VERSION=$PYTHON_VERSION
ARG N_JOBS=8
ENV N_JOBS=$N_JOBS
ENV MAKEFLAGS=-j${N_JOBS}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y --no-install-recommends --allow-unauthenticated apt-utils
RUN apt-get install -y --no-install-recommends --allow-unauthenticated \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    man-db \
    autoconf \
    pkg-config \
    unzip \
    libffi-dev \
    software-properties-common \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev

RUN mkdir /fever
WORKDIR /fever

COPY make_replacement.sh /usr/local/bin/make
RUN bash -c 'chmod +x /usr/local/bin/make'

COPY environment.yml /fever/
COPY requirements.txt /fever/
#RUN mamba install --yes python=$PYTHON_VERSION 		
RUN mamba create --yes -n fever python=$PYTHON_VERSION

SHELL ["bash", "-lc"]
ENV PATH /opt/conda/envs/fever/bin:$PATH
ENV TARGETPLATFORM $TARGETPLATFORM

# RUN /bin/bash -c "conda activate fever"
# RUN conda activate fever


# RUN pip install -r requirements.txt

# RUN python -c "import nltk; nltk.download('punkt')"

# RUN mkdir -pv src
# RUN mkdir -pv configs

# ADD src src
# ADD configs configs

# ADD predict.sh .

# ENV PYTHONPATH src
# ENV FLASK_APP sample_application:my_sample_fever

# #ENTRYPOINT ["/bin/bash","-c"]
# CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "sample_application:my_sample_fever"]
