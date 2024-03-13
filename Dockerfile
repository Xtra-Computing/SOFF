FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Requirements, Dependencies for cv2, and utilities
RUN apt update \
    && apt upgrade --yes \
    && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata \
    && apt install --yes --force-yes \
        sudo git curl jq lsof tmux openssh-client openssh-server openssh-sftp-server \
        python3.9-full python3.9-dev python3-pip cmake \
        libglib2.0-0 libsm6 libxrender1 libxext6 libglew-dev \
        libevent-dev libncurses-dev \
        iproute2 net-tools iputils-ping bindfs \
        vim htop

# Extra setups
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3

# Setup ssh and users
RUN ssh-keygen -A \
    && mkdir /run/sshd \
    && echo '   StrictHostKeyChecking no' >> /etc/ssh/ssh_config \
    && echo '   PermitRootLogin yes' >> /etc/ssh/sshd_config \
    && useradd -m user \
    && usermod --password $(echo 123 | openssl passwd -1 -stdin) root \
    && usermod --password $(echo 123 | openssl passwd -1 -stdin) user \
    && chsh -s /bin/bash user \
    && echo 'user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers \
    && su user -c 'ssh-keygen -t rsa -N "" -f  /home/user/.ssh/id_rsa <<<y' \
    && su user -c 'cat /home/user/.ssh/id_rsa.pub > /home/user/.ssh/authorized_keys' \
    && cp -r /home/user/.ssh /root/

USER user
WORKDIR /home/user

# Setup tmux
RUN curl -L https://github.com/tmux/tmux/releases/download/3.2a/tmux-3.2a.tar.gz -o tmux-3.2a.tar.gz
RUN tar -xvzf tmux-3.2a.tar.gz && rm tmux-3.2a.tar.gz
WORKDIR /home/user/tmux-3.2a
RUN ./configure --prefix=/usr && make -j8 && sudo make install
WORKDIR /home/user
RUN rm -rf tmux-3.2a

# Code related ----------------------------------------------------------------
WORKDIR /home/user/code

# Install python requirements
COPY requirements.txt /home/user/code/
RUN sudo python3.9 -m pip install -r requirements.txt

# gpustat needs to be installed as user
RUN sudo python3.9 -m pip install gpustat

RUN mkdir /tmp/soff

# Copy code
COPY --chown=user:user res/.tmux.conf res/.tmux.conf.local /home/user/

CMD ["sudo", "/usr/sbin/sshd", "-D"]
