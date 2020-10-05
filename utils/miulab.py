"""
Copy file (including metric) from MiuLab:

	https://github.com/MiuLab/SlotGated-SLU
"""
import os


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(ss, correct_slots, pred_slots, args):
    correctChunk = {}
    correctChunkCnt = 0.0
    foundCorrect = {}
    foundCorrectCnt = 0.0
    foundPred = {}
    foundPredCnt = 0.0
    correctTags = 0.0
    tokenCount = 0.0
    if ss is None:
        ss = [["UNK" for s in xx] for xx in correct_slots]
        ffile = "eval.txt"
    else:
        ffile = "eval_all.txt"
    with open(os.path.join(args.save_dir, ffile), "w", encoding="utf8") as writer:
        for correct_slot, pred_slot, tokens in zip(correct_slots, pred_slots, ss):
            inCorrect = False
            lastCorrectTag = 'O'
            lastCorrectType = ''
            lastPredTag = 'O'
            lastPredType = ''
            for c, p, token in zip(correct_slot, pred_slot, tokens):
                writer.writelines("{}\t{}\t{}\t{}\t{}\n".format(token, "n", "O", c, p))
                correctTag, correctType = __splitTagType(c)
                predTag, predType = __splitTagType(p)

                if inCorrect == True:
                    if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                            __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                            (lastCorrectType == lastPredType):
                        inCorrect = False
                        correctChunkCnt += 1.0
                        if lastCorrectType in correctChunk:
                            correctChunk[lastCorrectType] += 1.0
                        else:
                            correctChunk[lastCorrectType] = 1.0
                    elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                            __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                            (correctType != predType):
                        inCorrect = False

                if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (correctType == predType):
                    inCorrect = True

                if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                    foundCorrectCnt += 1
                    if correctType in foundCorrect:
                        foundCorrect[correctType] += 1.0
                    else:
                        foundCorrect[correctType] = 1.0

                if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                    foundPredCnt += 1.0
                    if predType in foundPred:
                        foundPred[predType] += 1.0
                    else:
                        foundPred[predType] = 1.0

                if correctTag == predTag and correctType == predType:
                    correctTags += 1.0

                tokenCount += 1.0

                lastCorrectTag = correctTag
                lastCorrectType = correctType
                lastPredTag = predTag
                lastPredType = predType

            if inCorrect == True:
                correctChunkCnt += 1.0
                if lastCorrectType in correctChunk:
                    correctChunk[lastCorrectType] += 1.0
                else:
                    correctChunk[lastCorrectType] = 1.0

    if foundPredCnt > 0:
        precision = 1.0 * correctChunkCnt / foundPredCnt
    else:
        precision = 0
    if foundCorrectCnt > 0:
        recall = 1.0 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0
    if (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    out = os.popen('perl ./utils/conlleval.pl -d \"\\t\" < {}'.format(os.path.join(args.save_dir, ffile))).readlines()
    f1 = float(out[1][out[1].find("FB1:") + 4:-1].replace(" ", "")) / 100

    return f1, precision, recall
