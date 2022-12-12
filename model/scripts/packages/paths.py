import os
from pathlib import Path

class ODISPaths():
    def __init__(self) -> None:

        # Root directory of the project
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent # Directorio base de la carpeta del proyecto. La que contiene a todos los archivos
        
        # Data folder
        self.DATA = os.path.join(self.BASE_DIR, 'data') 

        # Upload folder, where the user's images are going to be uploaded
        self.UPLOAD_FOLDER = os.path.join(self.DATA, 'uploads') 

        self.PREDICTIONS = os.path.join(self.DATA, 'predictions') 

        self.WEIGHTS = os.path.join(self.DATA, 'weights')
