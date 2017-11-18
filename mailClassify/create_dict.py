import MeCab
import pprint, sys, re
import csv
import codecs
import collections, itertools
import pickle

#
# Utilities for Debug
#
pp = pprint.PrettyPrinter(indent=4, width=80)

def myprint(str):
    print(str.encode('cp932','replace').decode('cp932'))

tagger = MeCab.Tagger("-Owakati")


def tokenizeCsvrow(csvrow):
    l = tagger.parse(csvrow[2]).replace("  "," ").split(" ")
    l = [ s.strip() for s in l]
    wordList = l
    l = tagger.parse(csvrow[3]).replace("  "," ").split(" ")
    l = [ s.strip() for s in l]
    wordList.extend(l)
    return wordList

def getAllList(csvName):
    # read csv
    csvfile = codecs.open(csvName, 'r', encoding="utf-8")
    csvReader = csv.reader(csvfile)
    csvData = [ row for row in csvReader ]
    csvfile.close()

    # get word List
    wordList = []
    for row in csvData:
        #print("###" + ', '.join(row))
        l = tokenizeCsvrow(row)
        wordList.extend(l)

    # get label list
    labelList = [ c[0] for c in csvData ]

    return wordList, labelList




#
# 
#
def createDict(csvName, max_freq=1, min_freq=0):

    print("reading all data from csv " + csvName)
    wordList, labelList = getAllList(csvName)
    #pp.pprint(wordList)

    print("creating dictionary")
    # word count
    wordCount = collections.Counter(wordList)
    # get dict
    l = [ w  for w, c in wordCount.items() if c <= max_freq * len(wordCount) and c >= min_freq * len(wordCount) ]
    wordDict = { c:i for i, c in enumerate(l) }
    print({ c:wordDict[c] for c in list(wordDict.keys())[0:5] })
    print("dictionary size: " + str(len(wordDict)))

    # label dict
    labels = collections.Counter(labelList).keys()
    labelDict = { c:i for i, c in enumerate(labels) }
    pp.pprint(labelDict)

    # debug out
    csvfile = codecs.open('wordCount.csv', 'w', encoding="utf-8")
    csv.writer(csvfile).writerows([ [w,c] for w, c in wordCount.items() ])
    csvfile.close()
    csvfile = codecs.open('wordDic.csv', 'w', encoding="utf-8")
    csv.writer(csvfile).writerows([ [i,w] for w, i in wordDict.items() ])
    csvfile.close()

    return wordDict, labelDict

#
#
#
def vectorize(csvrow, wordDict, labelDict):
    words = tokenizeCsvrow(csvrow)
    wordIds = [ wordDict[w] for w in words if w in wordDict ]
    wordVec = [0] * len(wordDict)
    for id in wordIds:
        wordVec[id] += 1

    labelId = labelDict[csvrow[0]]
    labelVec = [0] * len(labelDict)
    labelVec[labelId] = 1

    return wordVec, labelVec

if __name__ == "__main__":
    wordDict = createDict('./data0.csv')
    with open('wordDict.pickle', mode='wb') as f:
       pickle.dump(wordDict, f)
    #with open('wordDict.pickle', mode='rb') as f:
    #   wordDict = pickle.load(f)


