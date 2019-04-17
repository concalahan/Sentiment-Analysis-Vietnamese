import os
import json, codecs

def createDirectory(directory):
    if os.path.exists(directory):
        print(directory + ' is already created !')
    else:
        print('Create directory : ' + directory)
        os.makedirs(directory)

def writeTextFile(path, data):
    f = codecs.open(path, "w", "utf-8")
    f.write(data)
    f.close()

def appendTextFile(path, data):
    if not os.path.isfile(path):
        f = open(path, 'w+')
        f.write('')
    with open(path, 'a') as file:
        file.write(data + '\n')

def eraseTextFile(path):
    with open(path, 'w'):
        pass