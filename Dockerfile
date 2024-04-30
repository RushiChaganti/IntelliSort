FROM python:3.9-slim

WORKDIR /app

COPY . /app


RUN pip install virtualenv

RUN python -m venv myenv

SHELL ["bash", "-c"]
RUN source myenv/bin/activate

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "st.py"]
