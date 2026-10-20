# Stack-Tox prediction environment.
# Models were trained on Kaggle with Python 3.12.13, scikit-learn 1.6.1, numpy 2.0.2.
# If the exact patch image is available, build with:  --build-arg PYTHON_VERSION=3.12.13
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by LightGBM / scikit-learn; libxrender1 by RDKit's shared libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY test_prediction.py .

# Fail the build early if the pinned stack does not import
RUN python -c "import sklearn, numpy, lightgbm, rdkit, pandas, joblib; print('sklearn', sklearn.__version__, '| numpy', numpy.__version__)"

# Your files (sample_smiles.csv, saved_models/) are mounted here at run time;
# predictions are written back to the same folder.
WORKDIR /data
ENTRYPOINT ["python", "/app/test_prediction.py"]
