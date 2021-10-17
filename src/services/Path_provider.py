import os
from pathlib import Path
from typing import Union
from datetime import datetime


class GLobalPathProvider:
    def __init__(self, file_name: Union[str, None]) -> None:
        f_name = "_" + file_name
        __date = datetime.now().strftime("%m-%Y/%d/ %H:%M:%S")
        self.path = str(Path(__file__).parent.parent.parent)
        self.log_path = self.path + "/logs/" + __date + f_name
        os.makedirs(self.log_path)
