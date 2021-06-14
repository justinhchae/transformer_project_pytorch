from src import utils, bert_classifier
from src.utils import constants
import torch
import pandas as pd
import numpy as np
import os
import random
# force all caching to a local cache folder
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = constants.CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = constants.CACHE_PATH
torch.hub.set_dir(constants.CACHE_PATH)
# take care of a warning with tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == '__main__':
    # run config scripts to make folders and things
    device = utils.config.run()
    seed = 42

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch_corpora_names = ['ag_news', 'imdb', 'yelp_polarity', 'sogou_news', 'dbpedia', 'yahoo_answers']

    results = []
    # run the network for the bert sequence classification model
    for torch_corpora_name in torch_corpora_names:
        bert_classifier.fit(torch_corpora_name=torch_corpora_name, device=device, do_break_testing=False)
        # results.append(result)

    print("=" * 20, "Final Total Experiment Summary", "=" * 20)
    df = pd.concat(pd.DataFrame(i) for i in results)
    print(df)
    results_filepath = os.sep.join([utils.constants.DATA_PATH, 'results.csv'])
    df.to_csv(results_filepath, index=False)
