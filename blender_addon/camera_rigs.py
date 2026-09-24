"""Camera-library locations, independent of the current .blend or installation."""
import os
import sys
from pathlib import Path


def bundled_dir():
    return Path(__file__).parent / 'presets' / 'camera_rigs'


def default_user_dir():
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
    return base / 'SplatGen' / 'Camera Rigs'
