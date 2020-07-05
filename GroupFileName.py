#python -m pip install fuzzywuzzy
#from fuzzywuzzy import fuzz
#python -m pip install python-Levenshtein
import Levenshtein

combined_list = ['rakesh', 'zakesh', 'bikash', 'zikash', 'goldman LLC', 'oldman LLC', 'file01', 'file02', 'file03', 
                 'fileWithDiffrentName']
combined_list.append('bakesh')
combined_list.extend(['name_is_here_100_20131204.txt',        # should accept
            'name_is_here_100_20131204.txt.NEW',    # should reject
            'other_name.txt',  #rejected!
            'name_is_44ere_100_20131204.txt',
            'name_is_here_100_2013120499.txt', 
            'name_is_here_100_something_2013120499.txt',
            'name_is_here_100_something_20131204.txt'])
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




from abc import ABCMeta, abstractmethod, abstractproperty
import re

class MatchClass():
    __metaclass__=ABCMeta

    #Совпадение
    @abstractmethod
    def match():
        return NotImplementedError("method match must be implemented")
        
class Template(MatchClass):
    def __init__(self, pattern):
        self.pattern = pattern
        self.regexe = re.compile(pattern)
        self.group = list() 

    def match(self, listOfStrings):
        group = [item for item in listOfStrings if self.regexe.match(item)]
        if not group:
            return listOfStrings
        else:
            return [item for item in listOfStrings if item not in group] 
    
assert issubclass(Template, MatchClass)
#assert ininstance(Template(), MatchClass)

class LevenshteinMatch(MatchClass):
    def __init__(self, distance):
        self.__distance = distance
        self.group = list() 

    def match(self, listOfStrings):
        #for name in listOfStrings:
        #    for g in self.group:
        #        if all(Levenshtein.distance(name, w) < 5 for w in g):
        #            g.append(name)
        #            break
        #    else:
        #        group.append([name, ])

        group = [item for item in listOfStrings if Levenshtein.distance(item, w) < self.__distance for w in self.group]
        if not self.group:
            [item for item in listOfStrings if item not in group]
        else:
            return listOfStrings
    
assert issubclass(LevenshteinMatch, MatchClass)
#assert ininstance(LevenshteinMatch(), MatchClass)



matches = list()
matches.append(Template(r'^name_is_here_(\d+)_20131204.txt$'))
matches.append(Template(r'^(\d+)_(\w+)$'))
matches.append(LevenshteinMatch(5))

tmp_file_list = combined_list
for mtch in matches:
    print('tmp_file_list:', tmp_file_list)
    tmp_file_list = mtch.match(tmp_file_list)
grs.append([name, ])
print('output groups:', grs)

#template = re.compile()


#regexes = [
    # your regexes here
#    re.compile(r'^name_is_here_(\d+)_20131204.txt$'),
#    re.compile(r'^(\d+)_(\w+)$'),
#    re.compile(...),
#    re.compile(...),
#]

mystring = 'hi'

#if any(regex.match(mystring) for regex in regexes):
#    print('Some regex matched!')

testList = ['name_is_here_100_20131204.txt',        # should accept
            'name_is_here_100_20131204.txt.NEW',    # should reject
            'other_name.txt',  #rejected!
            'name_is_44ere_100_20131204.txt',
            'name_is_here_100_2013120499.txt', 
            'name_is_here_100_something_2013120499.txt',
            'name_is_here_100_something_20131204.txt']  

#acceptList = [item for item in testList if template.match(item)]




def find(scenario):
    begin  = '[a-z_]+100_' # any combinations of chars and underscores followd by 100
    end = '_[0-9]{8}.txt$' #exactly eight digits followed by .txt at the end
    pattern = re.compile("".join([begin,scenario,end]))
    result = []
    for word in testList:
        if pattern.match(word):
            result.append(word)

    return result

find('something') # returns ['name_is_here_100_something_20131204.txt']