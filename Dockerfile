FROM python:3.11-slim

WORKDIR /app

RUN pip install streamlit pandas numpy

COPY simple_app.py app.py

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
