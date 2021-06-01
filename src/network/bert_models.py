import torch
from transformers import DistilBertConfig, DistilBertModel


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        configuration = DistilBertConfig()
        self.dim = configuration.dim
        self.bert = DistilBertModel(configuration)
        self.relu = torch.nn.ReLU()
        self.linearOut = torch.nn.Linear(self.dim, 1)

    def forward(self, x, att):
        bert_output = self.bert(x, attention_mask=att)

        output = self.linearOut(bert_output.last_hidden_state)

        return output
