FROM python:3.9.5-buster
RUN apt-get update && apt-get install -y \
	python-numpy \
	python-scipy \
	python-dev \
	python-nose \
	g++ \
	libblas-dev \
	git \
	curl

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip --no-cache-dir install -U spacy
RUN pip --no-cache-dir install scispacy

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
	Cython \
	wheel \
	jupyterlab

# COPY en_core_sci_scibert-0.4.0.tar.gz /home/
# WORKDIR /home/
# RUN pip --no-cache-dir install en_core_sci_scibert-0.4.0.tar.gz
RUN pip --no-cache-dir install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz

#Jupyter notebook related configs
COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888