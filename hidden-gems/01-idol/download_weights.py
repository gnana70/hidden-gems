"""Download everything Idol needs, once, and check access to the two gated encoders first.

  python download_weights.py
"""
import os
import sys
import time

os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

from huggingface_hub import auth_check, get_token, snapshot_download, whoami
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

GATED = {
    'facebook/dinov3-vith16plus-pretrain-lvd1689m': 'manual approval by Meta (can take a while)',
    'black-forest-labs/FLUX.2-dev': 'click to accept (instant); only the VAE is downloaded',
}

if get_token() is None:
    sys.exit('No Hugging Face token found. Copy .env.example to .env in this folder and put your Read token there (from '
             'https://huggingface.co/settings/tokens).')
print('Hugging Face account:', whoami()['name'])
missing = []
for repo, note in GATED.items():
    try:
        auth_check(repo)
        print(f'  ok      {repo}')
    except (GatedRepoError, RepositoryNotFoundError):
        print(f'  BLOCKED {repo}  -> accept the terms at https://huggingface.co/{repo}  ({note})')
        missing.append(repo)
if missing:
    sys.exit('Access to the gated encoders is missing (see above). Accept the terms, wait for approval, then re-run.')

print('Downloading (about 6.8 GB the first time): Idol weights 3.1 GB, DINOv3 ViT-H+ 3.4 GB, FLUX.2 VAE 0.3 GB')
def fetch(repo, patterns):
    for attempt in range(1, 6):   # resumes where it stopped
        try:
            return snapshot_download(repo, allow_patterns=patterns)
        except Exception as e:  # noqa: BLE001
            print(f'   {repo}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]}), retrying in 20 s')
            time.sleep(20)
    sys.exit(f'Could not download {repo}. Check the connection and run the same command again (finished files are kept).')


fetch('speridlabs/idol', ['config.json', 'denoiser.safetensors', 'decoder.safetensors', 'decoder_gs.safetensors',
                          'birefnet.safetensors'])
fetch('facebook/dinov3-vith16plus-pretrain-lvd1689m', ['*.json', '*.safetensors'])
fetch('black-forest-labs/FLUX.2-dev', ['vae/*'])
print('All weights are in the Hugging Face cache.')
