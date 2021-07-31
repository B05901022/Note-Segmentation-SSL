# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:16:37 2021

@author: Austin Hsu
"""

from google_drive_downloader import GoogleDriveDownloader as gdd
from zipfile import ZipFile

DATA_FILE_ID = "1UPcvK1favpIoiYaL8qNqgbNbI3jn3lm3"

def download_from_googledrive(file_id: str = DATA_FILE_ID, download_path: str = "../data.zip"):
    gdd.download_file_from_google_drive(file_id, download_path)
    print('Download complete.')
    return

def unzip_file(file_name: str = '../data.zip', file_dest: str = '../'):
    file = ZipFile(file_name)
    file.extractall(file_dest)
    print('Extraction complete.')
    return

if __name__ == "__main__":
    download_from_googledrive()
    unzip_file()