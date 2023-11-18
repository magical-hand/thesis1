from flair.data import Sentence
str_text='i have a pen'
str_list=str_text.split()
sentence=Sentence(str_list)
print(sentence)

print(str_list)