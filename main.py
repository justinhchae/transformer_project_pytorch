from src import util, network
import torch

torch.hub.set_dir(util.constants.CACHE_PATH)


if __name__ == '__main__':
    # run config scripts to make folders and things
    device = util.config.run()

    # run validator for data structures and model
    network.run_network(device)

    # run the jiant validator script
    # util.validate_jiant()

