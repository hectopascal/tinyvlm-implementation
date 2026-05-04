python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
curl -LsSf https://hf.co/cli/install.sh | bash

hf download liuhaotian/LLaVA-Pretrain --local-dir /tmp/data --repo-type dataset
unzip /tmp/data/images.zip -d /tmp/data/images