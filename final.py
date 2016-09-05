import nltk
import pickle
import math
import re
from operator import itemgetter

stopWords = nltk.corpus.stopwords.words('english')

def save_object(obj, filename):
    '''
    Save object using pickle

    :param obj: Python object, object to be saved
    :param filename: str, name of file to be saved
    '''

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, -1)

def load_object(filename):
    '''
    Load object in same directory using name

    :param filename: str, name of file to be loaded
    :return: Python object, object to be loaded
    '''

    file = open(filename, 'rb')

    return pickle.load(file)

def stop_words(text):
    '''
    Remove "Stop Words" as defined by NLTK library

    :param text: str, to have stop words removed
    :return: str, without stop words
    '''
    string = ' '.join([word for word in text.split() if word not in stopWords])
    return string

def review_text(str):
    '''
    Given a string remove symbols that do not contribute to meaning of string
    Returns string split into lines by meaning indicative punctuation
    (periods, exclamations points, question makes, semi-colons etc.)

    :param str: str, iven string
    :return: str, string without punctuation and with \n
    '''
    lowercase = str.lower()
    stopWords = stop_words(lowercase)
    periods = re.sub("\.\.*", '\n', stopWords)
    commas = periods.replace(",", "\n")
    x = commas.replace("&quot;","")
    y = x.replace("&gt;", "")
    z = y.replace("&lt;","")

    dict = {"\"":"",
            "\'":"",
            "!":"\n",
            ";":"\n",
            ",":"",
            "\u2026":"",
            "?":"\n",
            "-":" ",
            "1": "",
            "2": "",
            "3": "",
            "4": "",
            "5": "",
            "6": "",
            "7": "",
            "8": "",
            "9": "",
            "0": "",
            "(": "",
            ")": "",
            ":":"",
            "&amp":"",
            "$":"",
            "#":"",
            "*":"",
            "\\ud":"",
            "/": " ",
            "+": "",
            "%":"",
            "@":"",
            "~":"",
            "`":"",
            "^":"",
            "&":"",
            "_":"",
            }

    for i, j in dict.items():
        z = z.replace(i, j)

    return z

def create_n_gram_list(num, file):
    '''
    Create a list of all num-degree grams within a given file

    :param num: int, degree of n-gram being created
    :param file: str, name of text file used to create n-grams
    :return: list(tuple(str)), list of num-degree tuples containing strings from text doc
    '''

    ngramList = []
    with open(file, 'r') as text:
        for line in text:
            splitLine = line.split()
            lineList = list(nltk.ngrams(splitLine, num))
            ngramList += lineList
    text.close()

    return ngramList

def create_n_gram_dict(gramList):
    '''
    Create a dictionary that counts the number of grams within a list

    :param gramList: list(tuple(str)), list of tuples
    :return: dict(tuple(str):int), dictionary of tuples and their count within gramList
    '''

    dict = {}
    for j in range(0, len(gramList)):
        if gramList[j] not in dict:
            dict[gramList[j]] = 1
        else:
            dict[gramList[j]] += 1

    return dict


def create_n_gram_list_str(num, string):
    '''
    Create n-gram list object given a string

    :param num: int, degree of n-gram
    :param string: str, string to be converted to n-gram object
    :return: list of n-degree tuples
    '''

    split = string.split()
    ngramList = list(nltk.ngrams(split, num))

    return ngramList

def create_str_list_object(string):
    '''
    Create object that contains all of the different degree n-grams

    :param string: str, string to be converted object
    :return: list(list(tuples)), list of list of n-degree tuples
    '''
    obj = [0]
    for i in range(1,10):
        lst = create_n_gram_list_str(i,string)
        obj.append(lst)
    return obj


def create_language_model2(file):
    dictList = [0]
    for i in range(1, 10):
        lst = create_n_gram_list(i, file)
        dict = create_n_gram_dict(lst)
        dictList.append(dict)

    return dictList

def create_big_object():
    cats = ["Appliances", "Heating, Venting & Cooling", "Tools & Hardware", "Decor",
            "Flooring", "Lighting & Ceiling Fans", "Outdoors", "Bath", "Doors & Windows", "Paint",
            "Storage & Organization", "Plumbing", "Building Materials"]

    dict = {}
    for i in cats:
        dict[i] = create_language_model2(i + " v2")

    save_object(dict, "Large Model 2")

def test_entropy_string(string, model):
    #RETURN LIST OF ENTROPIES FOR 2,3,4,5,6,7,8,9 GRAM ON FOR MODEL
    #test: Appliances Test List
    #model: Bath Dict Model

    listObj = create_str_list_object(string)
    modelDict = load_object(model)
    results = []

    for i in range(3,7):
        entropy = 0.0
        for x in listObj[i]:
            ans = recursive_smoothing(x, modelDict, i)
            prob = math.log2(ans)
            entropy += prob

        final = (entropy * -1) / len(listObj)
        results.append(str(final))

    return results



def test_entropy_final(obj, model, num, category):
    #RETURN LIST OF ENTROPIES FOR 2,3,4,5,6,7,8,9 GRAM ON FOR MODEL
    #test: Appliances Test List
    #model: Bath Dict Model

    entropy = 0.0
    for x in obj:
        ans = recursive_smoothing(x, model, num)
        if ans == -1:
            continue
        prob = math.log2(ans)
        entropy += prob

    final = (entropy * -1) / len(obj)
    tup = (final, str(category))

    return tup

def categorize_test(review):
    cats = ["Appliances", "Heating, Venting & Cooling", "Tools & Hardware", "Decor",
            "Flooring", "Lighting & Ceiling Fans", "Outdoors", "Bath", "Doors & Windows", "Paint",
            "Storage & Organization", "Plumbing", "Building Materials"]

    file = open("final test " + str(7), "a+")
    file.write(review_text(review))
    revisedStr = create_str_list_object(review_text(review))[7]
    file.write("revised string:   " + str(revisedStr))
    lst = []

    for i in cats:
        result = test_entropy_final(revisedStr, i, 7)
        lst.append(result)

    print(str(lst) + " " + str(type(lst)))
    final = sorted(lst, key=itemgetter(0))
    print("\n" + str(final) + " " + str(type(final)))
    file.write("\n\nfinal list:   " + str(final))
    file.close()
    return final[16][1] + " " + final[15][1] + " " + final[14][1] + " " + final[13][1]

def categorize(review):
    cats = ["Appliances", "Heating, Venting & Cooling", "Tools & Hardware", "Decor",
            "Flooring", "Lighting & Ceiling Fans", "Outdoors", "Bath", "Doors & Windows", "Paint",
            "Storage & Organization", "Plumbing", "Building Materials"]

    model = load_object("Large Model 2")
    revisedStr = create_str_list_object(review_text(review))[7]
    lst = []

    for i in cats:
        result = test_entropy_final(revisedStr, model[i], 7, i)
        lst.append(result)

    final = sorted(lst, key=itemgetter(0))
    x = []
    for i in range(0,3):
        x.append(final[i][1])

    return x

def recursive_smoothing(gram, model, num, r=0):
    #[i-r][x[-i+r:]] = num
    #[i-r+1][x[-i+r:-1]] = denom
    #r++ until no error or unigram
    if (r == num - 1):
        #return unigram(gram,model,num,r)
        return 1/10000
    try:
        numerator = model[num - r][gram[-num + r:]]
        denominator = model[num - (r+1)][gram[-num + r:-1]]
        return numerator/denominator
    except KeyError:
        r += 1
        return recursive_smoothing(gram, model, num, r)

def unigram(gram,model,num,r):
    file = open("unigram testing 4", "a+")
    try:
        numerator = model[num - r][gram[-num + r:]]
        denominator = sum(model[1].values())
        file.write(str(gram[-num + r:]) + "Num: " + str(numerator) + "Denom: " + str(denominator)+ "\n\n")
        file.close()
        return numerator/denominator

    except KeyError:
        return -1

def recursive_smoothing_test(gram, model, num, category, r=0):
    # [i-r][x[-i+r:]] = num
    # [i-r+1][x[-i+r:-1]] = denom
    # r++ until no error or unigram
    file = open("smoothing record final " + str(category), "a+")
    try:
        numerator = model[num - r][gram[-num + r:]]
        denominator = model[num - (r + 1)][gram[-num + r:-1]]
        file.write("Attempt #: " + str(r + 1) + str(gram[-num + r:]) + " Numerator= " + str(
            model[num - r][gram[-num + r:]]) + " " + str(gram[-num + r:-1]) + " Denominator = " + str(
            model[num - (r + 1)][gram[-num + r:-1]]) + "\n")
        file.close()
        return numerator / denominator
    except KeyError:
        if (r == num - 2):
            file.write("Attempt #: " + str(r + 1) + " Failed\n")
            file.close()
            return 1 / 10000
        else:
            r += 1
            file.close()
            return recursive_smoothing_test(gram, model, num, category, r)

    return


def final_testing_object(category, review):
    lst = load_object("Final Testing Object")
    obj = (category, review)
    lst.append(obj)
    save_object(lst, "Final Testing Object")


def final_test():
    lst = load_object("Final Testing Object")
    num = 0
    denom = 0
    for i in lst:
        ans = categorize(i[1])
        if i[0] in ans:
            num += 1
        else:
            file = open("Final Testing Errors 2", "a+")
            file.write(str(i) + "\n" + str(ans) + "\n\n\n")
            file.close()
        denom += 1

    print(str(num) + "/" + str(denom))

final_test()


#final_testing_object("Tools & Hardware", "If you are looking for a cordless circular saw then you cant go wrong with this. I had to take out the subfloor in my kitchen and this thing did the job with ease. Torquey little guy. 2 caveats: a) This thing loves to suck the life out of batteries so make sure you're using a 4 amp heavy duty battery with it or you will be sorely dissapointed if you're planning to make more than a couple of cuts... and b) this saw will NOT cut you any slack if youre not using a new(ish) blade. I recently switched to a Red Devil 40 tooth and i must say it works much better than the 18 tooth that comes with the saw.")
#save_object(lst, "Final Testing Object Backup")
#print(len(load_object("Final Testing Object")))
#print(str(create_str_list_object(review_text("Hello there54345)%*)(@#$ this is a test to see i54380934f everything works i am not sure if it will, but it is worth a shot. thanks for trying"))[9]))
#categorize("Samsung did not disappoint with this top of the line washing machine. The washing machine comes with endless features, including sleek exterior design with touch control panel, high efficiency, large bowl and soaking tub. Even storage for the drain hose and cord! Samsung did an excellent job with packaging, no damage during shipment to my home. Setup of the machine was very simple, detailed instructions are provided. The touch control panel is very user friendly and gives settings for water temperature, spin cycle and soil level. The machine offers more flexibility than any other machine that I have previously owned. Rather than just cold warm or hot, you additionally have eco warm and tap cold. You also have selections for additional features such as extra rinse, fabric softener, presoak, eco plus, super speed waterproof, etc. You can save the cycle that you use the most as “My Cycle” for a quick start on busy days. When you select different cycle settings, the time will vary and is displayed on the control panel. One huge bonus with this washing machine is the size of the tub. I am able to wash very large loads, including bedding, linens, etc. that would have never fit in my previous machine. I put the washing machine to the test with a very large load and the clothes came out very clean and fresh. They were fluffed with no wrinkles and not in a big wad. Having a family, the pre soak bowl for stains is a fantastic feature. You can treat stains, soak and they are dropped into the bowl and washed – what a time saver! Compared to my old washer it uses a small amount of water and yet cleans better. Amazing.The washer is very quiet and smooth during operation, you can barely hear it running. It could be placed in any area in your home and wouldn’t be a disturbance. This Samsung washing machine offers superior quality, sleek design and endless features making a great addition to your appliances - I highly recommend this product!")