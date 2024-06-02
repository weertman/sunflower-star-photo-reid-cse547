from src.dataset_creation_scripts.Other_species_create_time_unaware_dataset import other_species
from src.dataset_creation_scripts.Pycnopodia_helianthoides_create_time_unaware_dataset import pycnopodia_helianthoides
from src.dataset_creation_scripts.Reduced_pycnopodia_helianthoides_create_time_unaware_dataset import reduced_pycnopodia_helianthoides
import multiprocessing

if __name__ == '__main__':
    ## run both at the same time using multiprocessing

    processes = []
    processes.append(multiprocessing.Process(target=other_species))
    processes.append(multiprocessing.Process(target=pycnopodia_helianthoides))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    reduced_pycnopodia_helianthoides()

    print('Done!')