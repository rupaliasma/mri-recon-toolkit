# GPU image for training and evaluation. Runs on CPU too, just slowly.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/mrirecon

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && rm -rf /root/.cache

COPY scripts ./scripts

# Mount fastMRI at /data and an output folder at /runs, e.g.
#   docker run --gpus all -v /path/to/fastmri:/data -v $PWD/runs:/runs mrirecon \
#     train --train-dir /data/singlecoil_train --out /runs/pdunet
VOLUME ["/data", "/runs"]
ENTRYPOINT ["mrirecon"]
CMD ["--help"]
