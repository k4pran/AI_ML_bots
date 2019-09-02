import os
import itertools
import logging

from . import *
from .data_not_found_exception import DataNotFoundException

logging.basicConfig(level=logging.INFO)

if os.path.isdir(POKEMON_DATA_DIR):
    logging.info("Path {} validated".format(POKEMON_DATA_DIR))
else:
    logging.error("Pokemon data directory {} does not exist".format(POKEMON_DATA_DIR))
    raise DataNotFoundException


def get_pokemon_names():
    logging.info("Fetching pokemon names")
    return next(os.walk(POKEMON_DATA_DIR))[1]


def get_pokemon_files(pokemon_dir):
    target_dir = os.path.join(POKEMON_DATA_DIR, pokemon_dir)
    logging.info("Fetching pokemon files from {}".format(target_dir))
    return [file for file in os.listdir(target_dir)]


def get_all_pokemon_files():
    dir_names = get_pokemon_names()
    return list(itertools.chain.from_iterable([get_pokemon_files(dir_name) for dir_name in dir_names]))