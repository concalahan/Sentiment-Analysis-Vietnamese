import json

def readJson(filePath):
    entities = dict()

    with open(filePath, encoding="utf-8" ) as file:
        data = json.load(file)
        for key, values in data.items():
            entities[key] = values
    
    return entities