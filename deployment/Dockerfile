FROM python:3.11-slim

RUN pip install -U pip

WORKDIR /docker-app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt 

COPY ["flask_app.py", "xgb.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "flask_app:app"]

