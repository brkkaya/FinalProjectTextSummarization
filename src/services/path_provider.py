import os
from pathlib import Path
from typing import Union
from datetime import datetime


class GLobalPathProvider:
    def __init__(self, file_name: Union[str, None] = None) -> None:
        if file_name:
            f_name = file_name + ".log"
        else:
            f_name = "console.log"
        __date = datetime.now().strftime("%m-%Y/%d/%H:%M:%S")
        self.path = str(Path(__file__).parent.parent.parent)
        self.logs_path = self.path + "/logs/" + __date
        os.makedirs(self.logs_path, exist_ok=True)
        self.log_path = self.logs_path + "/" + f_name
