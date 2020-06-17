#python -m pip install fuzzywuzzy
#from fuzzywuzzy import fuzz
#python -m pip install python-Levenshtein
import Levenshtein

combined_list = ['rakesh', 'zakesh', 'bikash', 'zikash', 'goldman LLC', 'oldman LLC', 'file01', 'file02', 'file03', 
                 'fileWithDiffrentName']
combined_list.append('bakesh')
print('input names:', combined_list)

grs = list() # groups of names with distance > 
for name in combined_list:
    for g in grs:
        if all(Levenshtein.distance(name, w) < 5 for w in g):
        #if all(fuzz.ratio(name, w) > 80 for w in g):
            g.append(name)
            break
    else:
        grs.append([name, ])

print('output groups:', grs)
outlist = [el for g in grs for el in g]
print('output list:', outlist)