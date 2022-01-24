"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
import math
def viterbi_2(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        START = train[0][0]
        END = train[0][len(train[0]) - 1]
        result = []
        tags = {}
        tagPair = {}
        tagdict = {}
        wlist={}
        
        
        if train is None or test is None:
                return result

        #for every word and tag in train data 
        for sentence in train:
                n=len(sentence)
                for word, tag in sentence:
                        if tag not in tagdict:
                                tagdict[tag] = {word : 1}
                        else:        
                                tagdict[tag][word] = tagdict[tag].get(word, 0) + 1
                        tags[tag] = tags.get(tag, 0) + 1
                        #we need to extract hapax words from training data
                        if word not in wlist:
                                wlist[word]=[tag]
                        else:
                                wlist[word].append(tag)
                for i in range(n - 1):
                        tagPair[(sentence[i][1], sentence[i+1][1])] = 1+tagPair.get((sentence[i][1], sentence[i+1][1]), 0) 
        #collect hapax words
        j =0
        hapax={}
        for w in wlist:
                if len(wlist[w])==1:
                        tag = wlist[w][0]
                        hapax[tag] = hapax.get(tag,0)+1
                        j+=1

        for sentence in test:
                sen_len=len(sentence)
                list = [START, END]
                dict = {}
                c = 0.00001
                
                smooth_tag = -400
                word = sentence[1]

                #for every tag in test data 
                for tag in tags:
                        #compute hapax tag probabilities
                        taglen=len(tagdict[tag])
                        p=(hapax.get(tag, 0) + c) / (j + c * (len(hapax) + 1))
                        wordSmooth = math.log(p / (tags[tag] + 1+(len(tagdict[tag])*p)))
                        if word not in tagdict[tag] and (START[1], tag) not in tagPair:
                                dict[(1, tag)] = (wordSmooth + smooth_tag, START[1])
                        elif word not in tagdict[tag]:
                                dict[(1, tag)] = (wordSmooth + p+ math.log((tagPair[(START[1], tag)] ) / (tags[tag] + p * (len(tags) + 1))), START[1])
                        elif (START[1], tag) not in tagPair:
                                dict[(1, tag)] = math.log((p+tagdict[tag][word] ) / (tags[tag] + taglen*p + 1)) + smooth_tag, START[1]
                        else:
                                dict[(1, tag)] = math.log((p+tagdict[tag][word]) / (tags[tag] + taglen*p + 1)) + math.log((tagPair[(START[1], tag)] + p) / (tags[tag] + p * (len(tags) + 1))), START[1]
                for i in range(2, sen_len - 1):
                        for tag in tags:
                                taglen=len(tagdict[tag])
                                p=(hapax.get(tag, 0) + c) / (j + c * (1+len(hapax)))
                                wordSmooth = math.log(p / (tags[tag] + 1+ (taglen*c)))
                                maximum = -999
                                max_tag = tag
                                word = sentence[i]
                                
                                for t in tags:
                                        if word not in tagdict[tag] and (t, tag) not in tagPair:
                                                e = wordSmooth + smooth_tag
                                        elif word not in tagdict[tag]:
                                                e = wordSmooth + math.log((tagPair[(t, tag)] + p) / (tags[tag] + p * (1+len(tags))))
                                        elif (t, tag) not in tagPair:
                                                e = math.log((p+tagdict[tag][word]) / (tags[tag] + 1+taglen*p )) + smooth_tag
                                        else:
                                                e = math.log((p+tagdict[tag][word]) / (tags[tag] + 1+taglen*p )) + math.log((tagPair[(t, tag)] + p) / (tags[tag] + (len(tags)*p + 1)))
                                        if maximum < e+dict[(i-1, t)][0]  :
                                                maximum = dict[(i-1, t)][0] + e
                                                max_tag = t
                                dict[(i, tag)] = (maximum, max_tag)
                maximum = -999
                #find the maximum
                for tag in tags:
                        if maximum < dict[(sen_len- 2, tag)][0] :
                                maximum = dict[(sen_len - 2, tag)][0]
                                max_tag = tag
                list.insert(1, (sentence[sen_len- 2], max_tag))

                for i in reversed(range(2, sen_len- 1)):
                        max_tag = dict[(i, max_tag)][1]
                        list.insert(1, (sentence[i-1],max_tag))
                result.append(list)
        return result
