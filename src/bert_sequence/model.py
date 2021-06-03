import torch
from transformers import BertForSequenceClassification


class Model(torch.nn.Module):
    def __init__(self
                 , num_labels=2
                 , activation="gelu"
                 , bert_type="bert-base-uncased"
                 ):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_type
                                                                  , output_hidden_states=True
                                                                  , num_labels=num_labels
                                                                  , hidden_act=activation)

        self.hidden_size = self.bert.config.hidden_size
        # use if going with a regression scheme
        self.fc = torch.nn.Linear(self.hidden_size, 1)

    def get_CLS_embeddings(self, layer):
        """Reference: @neeraj2020bertlayers, https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert
        """
        return layer[:, 0, :].detach().numpy()

    def make_cls_embeddings(self, output, echo=False):
        """Reference: @neeraj2020bertlayers, https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert
        """
        # returning the bert hidden states as 3-tuple of context data
        hidden_states = output.hidden_states

        # Per @neeraj2020bertlayers: return just the CLS context for classification tasks
        cls_embeddings = torch.tensor([self.get_CLS_embeddings(hidden_states[i]) for i in range(12)])
        # cls embeddings: batch x 768-dim
        if echo:
            print('Hidden states as Embeddings:', len(hidden_states), type(hidden_states))
            print("cls embeddings:", len(cls_embeddings), type(cls_embeddings), cls_embeddings[0].shape)
        # return the embeddings for the CLS token representing the document embedding space
        return cls_embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):

        output = self.bert(input_ids=input_ids
                           , attention_mask=attention_mask
                           , token_type_ids=token_type_ids
                           , labels=labels
                           )

        return output
