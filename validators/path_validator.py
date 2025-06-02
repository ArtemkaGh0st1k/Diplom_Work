import os
from os.path import join, exists
from os import getcwd, mkdir


class PathValidator:
    """Проверяеть корректность путей к файлам и каталогам"""


    @staticmethod
    def check_requred_dirs():
        cur_dir = getcwd()

        req_dirs = ["data", "optimizers", "results", "utils", "validators"]
        for rd in req_dirs:
            temp_path = join(cur_dir, rd)
            if not exists(temp_path):
                mkdir(temp_path)
    


