import os
def create_if_not_exists(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)