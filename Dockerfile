FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-pip python3-tk

ENV MPLBACKEND=Agg

WORKDIR /app

# --- Dependency installation (cached unless setup.py changes) ---
COPY setup.py /app/setup.py
COPY README.md /app/README.md
RUN mkdir -p /app/psflearning && touch /app/psflearning/__init__.py
RUN pip install /app

# --- Source code copy + editable install (no-deps, fast) ---
COPY . /app
RUN pip install --no-deps -e /app

CMD ["python3", "test/demo/test_demo_bead_plotting.py"]
