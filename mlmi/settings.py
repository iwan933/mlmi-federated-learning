from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve(strict=True).parent.parent
CACHE_DIR = REPO_ROOT / 'cache'
RUN_DIR = REPO_ROOT / 'run'
CHECKPOINT_DIR = REPO_ROOT / 'checkpoints'
