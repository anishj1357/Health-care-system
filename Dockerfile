FROM python:3.8

ENV PYTHONUNBUFFERED 1

ENV HOME_DIR=/webapp

RUN mkdir -p ${HOME_DIR}

WORKDIR ${HOME_DIR}

COPY . ${HOME_DIR}

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ${HOME_DIR}/entrypoint.sh
