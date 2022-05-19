FROM rayproject/ray:latest

WORKDIR /home/ray/workspace

COPY requirements.txt /home/ray/workspace/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .