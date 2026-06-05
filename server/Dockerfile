FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --metrics-export-port=8080 --disable-usage-stats && serve run ray.yml
