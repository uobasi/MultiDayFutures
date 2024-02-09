FROM python:3.9.7

ENV APP_HOME /app
ENV TZ="America/New_York"
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

EXPOSE 8080

CMD python app.py
