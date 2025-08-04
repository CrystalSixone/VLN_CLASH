import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, BertModel

class MyRobertaTokenizer:
    def __init__(self, max_length=80, load_model=False, device='cuda'):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length
        self.padding = True
        if load_model:
            self.model = RobertaModel.from_pretrained('roberta-base').to(device)
        
    def text_token(self, text, max_length=None):
        max_length = max_length if max_length is not None else self.max_length
        
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length) # input_ids, attention_mask
        return encoded_input
    
    def text_embeddings(self, text):
        encoded_input = self.text_token(text)
        output = self.model(**encoded_input)
        return output
    
    def convert_tokens_to_string(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        return decoded_text

class MyBertTokenizer:
    def __init__(self, max_length=80, load_model=False, device='cuda', model_name='bert'):
        bert_tok_path = 'data/pretrained/bert-base-uncased'
        roberta_tok_path = 'data/pretrained/roberta'
        
        if model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(bert_tok_path)
        elif model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained(roberta_tok_path)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        self.max_length = max_length
        self.padding = True
        if load_model:
            if model_name == 'bert':
                self.model = BertModel.from_pretrained(bert_tok_path).to(device)
            elif model_name == 'roberta':
                self.model = RobertaModel.from_pretrained(roberta_tok_path).to(device)
        
    def text_token(self, text, max_length=None):
        max_length = max_length if max_length is not None else self.max_length
        
        encoded_input = self.tokenizer(
            text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length)
        return encoded_input
    
    def text_embeddings(self, text):
        encoded_input = self.text_token(text)
        output = self.model(**encoded_input)
        return output
    
    def convert_tokens_to_string(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        return decoded_text