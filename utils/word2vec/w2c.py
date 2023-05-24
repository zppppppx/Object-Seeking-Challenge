from gensim.models.word2vec import Word2Vec, KeyedVectors
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
            lines = [line.split('\"')[1].split(' ') for line in lines]
            # print(len(lines))

            allwords.extend(lines)
    # print(len(allwords))
    return allwords
    
def load_model(model_path):
    return Word2Vec.load(model_path)

def load_vector(vector_path):
    return KeyedVectors.load_word2vec_format(vector_path, )

def similarity(wv, tgt1, tgt2):
    return wv.similarity(tgt1, tgt2)


if __name__ == "__main__":
    # allwords = preprocess(r'F:\grad\quater3\ECE285\Object-Seeking-Challenge\utils\word2vec')
    # print(allwords)
    # wv = Word2Vec(allwords, vector_size=512)
    
    # wv.save('home_wv.model')
    # model = Word2Vec.load('home_wv.model')
    # print(similarity(wv, 'table', 'chair'))

    

    # print(similarity(wv, 'chair', 'table'))

    wv = load_vector(r"F:\grad\quater3\ECE285\Object-Seeking-Challenge\model.txt")
    print(similarity(wv, 'chair', 'table'))
    print(wv['chair'])
    import torch

    wvs = torch.tensor(wv['chair']).view(1, -1)
    wvs = wvs.repeat(1, 1)

    print(wvs.shape)