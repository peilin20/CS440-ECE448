"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
def viterbi_1(train, test):
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
        #with tag count we train words
        for sentence in train:
                for word, tag in sentence:
                        if tag not in tagdict:
                                tagdict[tag] = {word : 1}
                        else:        
                                tagdict[tag][word] = tagdict[tag].get(word, 0) + 1
                        tags[tag] = tags.get(tag, 0) + 1
                for i in range(len(sentence) - 1):
                        tagPair[(sentence[i][1], sentence[i+1][1])] = 1+tagPair.get((sentence[i][1], sentence[i+1][1]), 0) 
                
        for sentence in test:
                list = [START, END]
                dict = {}
                c = 0.00001
                word = sentence[1]
                smooth_tag = -400

                #for every tag in test data 
                for tag in tags:
                        #compute smooth probabilities
                        taglen=len(tagdict[tag])
                        wordSmooth = math.log(c / (tags[tag] + 1+(len(tagdict[tag])*c)))
                        if word not in tagdict[tag] and (START[1], tag) not in tagPair:
                                dict[(1, tag)] = (wordSmooth + smooth_tag, START[1])
                        elif word not in tagdict[tag]:
                                dict[(1, tag)] = (wordSmooth + math.log((tagPair[(START[1], tag)] + c) / (tags[tag] + c * (len(tags) + 1))), START[1])
                        elif (START[1], tag) not in tagPair:
                                dict[(1, tag)] = math.log((tagdict[tag][word] + c) / (tags[tag] + taglen*c + 1)) + smooth_tag, START[1]
                        else:
                                dict[(1, tag)] = math.log((tagdict[tag][word] + c) / (tags[tag] + taglen*c + 1)) + math.log((tagPair[(START[1], tag)] + c) / (tags[tag] + c * (len(tags) + 1))), START[1]
                for i in range(2, len(sentence) - 1):
                        for tag in tags:
                                wordSmooth = math.log(c / (tags[tag] + 1+ (len(tagdict[tag])*c)))
                                maximum = -999
                                max_tag = tag
                                word = sentence[i]
                                taglen=len(tagdict[tag])
                                for t in tags:
                                        if word not in tagdict[tag] and (t, tag) not in tagPair:
                                                e = wordSmooth + smooth_tag
                                        elif word not in tagdict[tag]:
                                                e = wordSmooth + math.log((tagPair[(t, tag)] + c) / (tags[tag] + c * (len(tags) + 1)))
                                        elif (t, tag) not in tagPair:
                                                e = math.log((tagdict[tag][word] + c) / (tags[tag] + taglen*c + 1)) + smooth_tag
                                        else:
                                                e = math.log((tagdict[tag][word] + c) / (tags[tag] + taglen*c + 1)) + math.log((tagPair[(t, tag)] + c) / (tags[tag] + (len(tags)*c + 1)))
                                        if maximum < e+dict[(i-1, t)][0]  :
                                                maximum = dict[(i-1, t)][0] + e
                                                max_tag = t
                                dict[(i, tag)] = (maximum, max_tag)
                maximum = -999
                #find the maximum tag 
                for tag in tags:
                        if maximum < dict[(len(sentence) - 2, tag)][0] :
                                maximum = dict[(len(sentence) - 2, tag)][0]
                                max_tag = tag
                list.insert(1, (sentence[len(sentence) - 2], max_tag))

                for i in reversed(range(2, len(sentence) - 1)):
                        max_tag = dict[(i, max_tag)][1]
                        list.insert(1, (sentence[i-1],max_tag))
                result.append(list)
        return result