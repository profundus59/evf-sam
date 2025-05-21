from transformers import AutoTokenizer
import ipdb

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ipdb.set_trace()
sequence = "Hello, how are you doing today? I hope you're having a great day!"
tokens = tokenizer(sequence)
print("Tokens:", tokens.tokens())

