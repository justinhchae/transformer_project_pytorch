from src import utils, bert_sequence
import torch
from pprint import pprint

# sets global path to torch.hub cache (for download and recall)
torch.hub.set_dir(utils.constants.CACHE_PATH)


if __name__ == '__main__':
    # run config scripts to make folders and things
    device = utils.config.run()

    torch_corpora_names = ["ag_news", 'dbpedia']

    results = []
    # run the network for the bert sequence classification model
    for torch_corpora_name in torch_corpora_names:
        result = bert_sequence.run.network(device, torch_corpora_name=torch_corpora_name, do_break_testing=True)
        results.append(result)

    pprint(results)

    # uncomment below to run the jiant validator script
    # util.validate_jiant()

