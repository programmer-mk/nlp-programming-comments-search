import os
import uuid
from requests.models import ProtocolError

def create_directory(directory_name):
    if((directory_name is None) == False and len(directory_name) > 0):
        full_directory_path = f'resources/{directory_name}'
        if not os.path.exists(full_directory_path):
            os.makedirs(full_directory_path)

def write_bytes_to_file(directory_name, file_name, content):
    if os.path.isdir(directory_name):
        full_file_path = f'{directory_name}/{file_name}'
        with open(full_file_path, "wb") as file_to_write:
            file_to_write.write(content)
    else:
        print(f'\n{directory_name} is not an existing directory!')

def get_file_extension(full_file_name):
    (file_name, file_extension) = os.path.splitext(full_file_name)
    return file_extension

def get_file_name(full_file_name):
    (file_name, file_extension) = os.path.splitext(full_file_name)
    return file_name

def append_guid_to_file_name(full_file_name):
    file_name = get_file_name(full_file_name)
    file_extension = get_file_extension(full_file_name)
    output = f'{file_name}_{uuid.uuid1()}{file_extension}'
    return output