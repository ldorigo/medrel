# See here for image contents: https://github.com/microsoft/vscode-dev-containers/blob/v0.163.0/containers/python-3/.devcontainer/base.Dockerfile
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Optional] Allow the vscode user to pip install globally w/o sudo
# ENV PIP_TARGET=/usr/local/pip-global
# ENV PYTHONPATH=${PIP_TARGET}:${PYTHONPATH}
# ENV PATH=${PIP_TARGET}/bin:${PATH}
# # Note User id of medrel user
# RUN if ! cat /etc/group | grep -e "^pip-global:" > /dev/null 2>&1; then groupadd -r pip-global; fi \
#     && usermod -a -G pip-global vscode \
#     && usermod -u 1003 vscode \
#     && umask 0002 && mkdir -p ${PIP_TARGET} \
#     && chown :pip-global ${PIP_TARGET} \
#     && ( [ ! -f "/etc/profile.d/00-restore-env.sh" ] || sed -i -e "s/export PATH=/export PATH=\/usr\/local\/pip-global:/" /etc/profile.d/00-restore-env.sh )

ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN apt-get update && apt-get install -y \
    python-numpy \
    python-scipy \
    python-dev \
    python-nose \
    g++ \
    libblas-dev \
    curl \
    git

# Get Rust
# RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# # jupyter needs permissions to this path
# RUN sudo mkdir /usr/local/share/jupyter
# RUN sudo chown -R vscode /usr/local/share/jupyter
USER vscode
COPY requirements.txt /home/requirements.txt
WORKDIR /home/
RUN pip --no-cache-dir install -r requirements.txt
# seperate because they run loooong
# RUN pip --no-cache-dir install -U spacy
# RUN pip --no-cache-dir install scispacy

# # Install other useful Python packages using pip
# # down here so it can be changed fatser
# RUN pip --no-cache-dir install --upgrade ipython && \
#     pip --no-cache-dir install \
#     Cython \
#     wheel \
#     jupyterlab

# COPY en_core_sci_scibert-0.4.0.tar.gz /home/
# WORKDIR /home/
# RUN pip --no-cache-dir install en_core_sci_scibert-0.4.0.tar.gz
# RUN pip --no-cache-dir install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz

#Jupyter notebook related configs
# COPY .devcontainer/jupyter_notebook_config.py /root/.jupyter/
# EXPOSE 8888