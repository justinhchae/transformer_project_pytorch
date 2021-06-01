import torch
from transformers import BertConfig


class Model(torch.nn.Module):
    def __init__(self, num_labels=2, bert_type="bert-base-uncased"):
        super().__init__()
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=True, num_labels=num_labels)

        self.bert = torch.hub.load('huggingface/pytorch-transformers'
                                   , 'modelForSequenceClassification'
                                   , bert_type
                                   , config=config)

        config = self.bert.config
        self.hidden_size = config.hidden_size
        self.fc = torch.nn.Linear(self.hidden_size, 1)

    def get_CLS_embedding(self, layer):
        """Reference: @neeraj2020bertlayers, https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert
        """
        return layer[:, 0, :].detach().numpy()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.bert(input_ids=input_ids
                                , attention_mask=attention_mask
                                , token_type_ids=token_type_ids
                                , labels=labels
                                )
        # == alternate way to extract targeted cls embeddings == #
        # returning the bert hidden states as 3-tuple of context data
        embeddings = output.hidden_states
        print('Embeddings:', len(embeddings), type(embeddings))
        # Per @neeraj2020bertlayers: return just the CLS context for classification tasks
        cls_embeddings = torch.tensor([self.get_CLS_embedding(embeddings[i]) for i in range(12)])
        # cls embeddings: batch x 768-dim
        print("cls embeddings:", len(cls_embeddings), type(cls_embeddings), cls_embeddings[0].shape)
        x = self.fc(cls_embeddings)
        print(x)

        return x
