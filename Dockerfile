FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

RUN pip install -U pip setuptools poetry

WORKDIR /home/user
COPY --chown=user ./pyproject.toml /home/user/pyproject.toml
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip install -r requirements.txt
