FROM python:3.10-slim-buster

WORKDIR /app 

COPY requirements.txt ./

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade sentence-transformers

RUN pip install --upgrade transformers

RUN python -c "import nltk; nltk.download('punkt')"

EXPOSE 80

COPY . .

CMD ["python","main - Copy.py"]