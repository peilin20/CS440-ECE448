"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        
        tags = dict()
        tagdict=dict()
        for sentence in train:
                for word in sentence:
                #for every single word in training data
                        if word[0] not in tagdict.keys():
                                tagdict[word[0]]=[[word[1]],[1]]
                        elif word[1] in tagdict[word[0]][0]:
                                tagdict[word[0]][1][tagdict[word[0]][0].index(word[1])] += 1
                        else:
                                #add current word and tag to dict
                                tagdict[word[0]][0].append(word[1])
                                tagdict[word[0]][1].append(1)
                        if word[1] in tags.keys():
                                tags[word[1]]+=1
                        else:
                                tags[word[1]]=1
        tagsCount = list(tags.values())
        tagsList=list(tags.keys())
        i=tagsCount.index(max(tagsCount))
        mostFrequent=tagsList[i]
        result = []
        for sentence in test:
                sentenceList=[]
                for word in sentence:
                        if word in tagdict.keys():
                                sentenceList.append((word, tagdict[word][0][tagdict[word][1].index(max(tagdict[word][1]))]))
                        else:
                                sentenceList.append((word, mostFrequent))
                result.append(sentenceList)
        return result