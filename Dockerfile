FROM python:3.10.12-bullseye

RUN useradd -ms /bin/bash devcontainer

WORKDIR /usr/src/app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python src/main.py
