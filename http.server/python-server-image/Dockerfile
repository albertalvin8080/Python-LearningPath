FROM python:3.12-slim

WORKDIR /app

COPY /content/* /app

EXPOSE 8080

CMD ["python", "-m", "http.server", "8080", "-b", "0.0.0.0"]