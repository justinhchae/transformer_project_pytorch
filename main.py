from src import util
import torch

torch.hub.set_dir(util.constants.CACHE_PATH)


if __name__ == '__main__':
    # run config scripts to make folders and things
    device = util.config.run()
    # bert case type can be "uncased" or others, see docs
    bert_case_type = 'cased'
    # set batch size (could be as high as 32 or so)
    batch_size = 4

    # make a tokenizer from HF library
    tokenizer = util.make_tokenizer(bert_case_type)

    util.validate_jiant()

