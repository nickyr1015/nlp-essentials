from elit_tokenizer import EnglishTokenizer

if __name__ == '__main__':
   text = 'Emory NLP is a research lab in Atlanta, GA. It was founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University.'
   tokenizer = EnglishTokenizer()
   sentence = tokenizer.decode(text)
   print(sentence.tokens)
   print(sentence.offsets)