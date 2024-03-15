import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 item_embedding_dim,
                 word_embedding_dim,
                 args):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.activate = nn.ReLU()
        self.pooler = nn.Linear(word_embedding_dim, item_embedding_dim)
        
    def forward(self, text):
        batch_size, num_words = text.shape
        num_words = num_words // 2
        text_ids = torch.narrow(text, 1, 0, num_words)
        text_attmask = torch.narrow(text, 1, num_words, num_words)

        hidden_states = self.bert_model(input_ids=text_ids, attention_mask=text_attmask)[0]
        # cls_after_pooler = self.activate(self.pooler(hidden_states[:, 0]))
        cls_after_pooler = self.pooler(hidden_states[:, 0]) 
        return cls_after_pooler

class TextEmbedding(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(TextEmbedding, self).__init__()
        self.args = args
        # we use the title of item with a fixed length.
        self.text_length = args.num_words_title * 2 # half for mask
        self.text_encoders = TextEncoder(bert_model, args.embedding_dim, args.word_embedding_dim, args)

    def forward(self, news):
        return self.text_encoders(torch.narrow(news, 1, 0, self.text_length))
