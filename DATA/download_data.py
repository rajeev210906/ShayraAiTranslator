# pip install wget

import wget

#hugging face url To Download These Data
# https://huggingface.co/facebook/m2m100_418M/tree/main

URL1 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json?download=true" #0.9kb
URL2 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/generation_config.json?download=true" #0.3kb
URL3 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/pytorch_model.bin?download=true" #1.94gb
URL4 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model?download=true" #2.42mb
URL5 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/special_tokens_map.json?download=true" #1.14kb
URL6 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/tokenizer_config.json?download=true" #0.3kb
URL7 = "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json?download=true" #3.8kb


#This Download Tool is completely Owned by Ranjan Coder And it's Free to use for everyone forever

response = wget.download(URL1, "config.json")
response = wget.download(URL2, "generation_config.json")
response = wget.download(URL3, "pytorch_model.bin")
response = wget.download(URL4, "sentencepiece.bpe.model")
response = wget.download(URL5, "special_tokens_map.json")
response = wget.download(URL6, "tokenizer_config.json")
response = wget.download(URL7, "vocab.json")