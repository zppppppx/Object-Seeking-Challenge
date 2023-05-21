from gensim.models import Word2Vec
from os import path
import os


def preprocess(directory):
    """
    Preprocess the text files in the directory, turn the files into training text file.
    """
    allwords = []
    for dir, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # print(filename)
            if("train" in filename or "txt" not in filename):
                continue
            file = path.join(dir, filename)
            with open(file, 'r') as fp:
                lines = fp.readlines()

            lines.pop(0)
            # print(lines)
            lines = [[line.split('\"')[1]] for line in lines]
            # print(len(lines))

            allwords.extend(lines)
    # print(len(allwords))
    return allwords
    
def load(model_path):
    return Word2Vec.load(model_path)


if __name__ == "__main__":
    allwords = preprocess(r'F:\grad\quater3\ECE285\Object-Seeking-Challenge\utils\word2vec')
    print(allwords)
    wv = Word2Vec(allwords, vector_size=512)
    wv.save('home_wv.model')