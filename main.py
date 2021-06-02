from src import utils, bert_sequence
import torch

# sets global path to torch.hub cache (for download and recall)
torch.hub.set_dir(utils.constants.CACHE_PATH)


if __name__ == '__main__':
    # run config scripts to make folders and things
    device = utils.config.run()

    # run the network for the bert sequence classification model
    bert_sequence.run.network(device)

    # uncomment below to run the jiant validator script
    # util.validate_jiant()

