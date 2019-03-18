from utils import *

def separatingParagraph(paragraph):
    entities = readJson('entities.json')
    # extend all the entities in json file 
    valuesList = list(item for valueList in entities.values() for item in valueList)
    # Result is a list of each sentence separated by "." and ","
    result = list()
    # Split comment with character "."
    splitComment = paragraph.split(".")
    for comment in splitComment:
        # countKey var to check if there are more than 2 values of entities or not
        countKey = 0
        for value in valuesList:
            if value in comment:
                countKey += 1
                if countKey == 2: break
        
        # If there is only one entities in the comment then there is no need to separate the comment
        if countKey == 1:
            result += [comment]
        # else if there is more than 2 entities then we should separate the comment with ","
        elif countKey == 2:
            result += [x for x in comment.split(",")]

    return result

def mergeEntity(comment):
    result = set()
    entities = readJson('entities.json')
    
    # Get all the sentence from the paragraph
    sentenceList = separatingParagraph(comment)

    for sentence in sentenceList:
        # FlagExist for the case the we set only the sentence with one entity
        # flagExist = False
        for key in entities:
            for value in entities[key]:
                if value in sentence.lower():
                    # print(key + " - " + value + " - " + sentence)
                    result.add((key,sentence))
                    # flagExist = True
                    break
            # if flagExist is True: break
    
    # Return a tuple: 
    # first is key ("PIN", "MANHINH" , ...)
    # second is the comment sentence
    return result

def mergeAttribute(mergeEntityResult):
    result = list()
    attributes = readJson('attributes.json')
    
    for x in mergeEntityResult:\
        # x[0] is key ("PIN", "MANHINH" , ...)
        entity = x[0]
        attrList = list()
        for attr in attributes[entity]:
            if attr in x[1].lower():
                attrList.append(attr)
        if attrList != []:
            result.append((entity,attrList,x[1]))
                
    return result

def simpleAnalyzeOntology(paragraph):
    mergeEntityResult = mergeEntity(paragraph)
    result = mergeAttribute(mergeEntityResult)

    return result

if __name__ == "__main__":
    # simpleAnalyzeComment()
    comment = """Máy dùng cũng rất oke.Con pin cực trâu, 50% pin lướt fb choi game liên tục tầm 5h. Hiệu năng cũng khá ổn, dù dùng con chip hơi thấp nhưng tối ưu tốt nên ít xảy ra giật lag, màn hình thì rất tốt luôn, mỗi tội cái camera cứ sao sao ấy. Chụp hình rất nhòe và ko sắc nét tí nào, ko biết là do máy mình lỗi hay j nhưng chụp 2 camera đều xấu, zoom lên thì chi tiết bị nhòe ko thấy j."""
    print(comment)
    for x in simpleAnalyzeOntology(comment):
        print(x)