import os

def create_directories(config):
    """
    Vérifie et crée les répertoires définis dans la configuration s'ils n'existent pas.

    :param config: Dictionnaire contenant les chemins des répertoires.
    """
    directories = [
        config["result"]["csv_directory"],
        config["result"]["jpg_directory"]
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)