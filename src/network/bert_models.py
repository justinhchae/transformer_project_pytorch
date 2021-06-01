import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = torch.hub.load('huggingface/pytorch-transformers'
                                   , 'model'
                                   , 'bert-base-uncased'
                                   , output_hidden_states=True)

        config = self.bert.config
        self.hidden_size = config.hidden_size
        self.fc = torch.nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids
                                , attention_mask=attention_mask
                                , token_type_ids=token_type_ids)
        # batch x input len x bert dim
        last_hidden_state = bert_output.last_hidden_state
        # TODO: apply
        return last_hidden_state
