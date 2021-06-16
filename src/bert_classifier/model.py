import torch
from transformers import RobertaForSequenceClassification


class Model(torch.nn.Module):
    def __init__(self
                 , num_labels
                 , pre_trained_str="roberta-base"
                 , layer_1_output_size=64
                 , dropout_percentage=.3
                 ):
        super().__init__()

        self.bert = RobertaForSequenceClassification.from_pretrained(pre_trained_str
                                                                     , output_hidden_states=True
                                                                     , num_labels=num_labels
                                                                     )
        self.config = self.bert.config
        self.parameters = self.bert.parameters
        self.save_pretrained = self.bert.save_pretrained

    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids=input_ids
                           , attention_mask=attention_mask
                           , labels=labels)

        return output
