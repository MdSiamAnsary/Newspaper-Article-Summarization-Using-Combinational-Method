rouge_one_precision = []
rouge_one_recall = []
rouge_one_fscore = []


def converting(str):
    st = str.split(' ')

    rof = st[2].strip(",")
    rouge_one_fscore.append(float(rof))
    rop = st[4].strip(",")
    rouge_one_precision.append(float(rop))
    ror = st[6].strip("},")
    rouge_one_recall.append(float(ror))


def sumci(x):
    res = 0
    for ii in x:
        res += ii
    return (res/len(x))

filename = "evaluation_result.txt"
lines = tuple(open(filename, 'r'))
#print(lines)
for line in lines:
    converting(line)

print("Implemented Rouge One F Score    : ", sumci(rouge_one_fscore ))
print("Implemented Rouge One Precision  : ", sumci(rouge_one_precision))
print("Implemented Rouge One Recall     : ", sumci(rouge_one_recall))