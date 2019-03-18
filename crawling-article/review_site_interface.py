from abc import ABCMeta, abstractmethod
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

class reviewSite():
    #Define Abtract Class
    __metaclass__ = ABCMeta

    _initUrl = ""
    
    _content = ""

    @classmethod
    def setInitUrl(self, initUrl):
        self._initUrl = initUrl

    @classmethod
    def getInitUrl(self):
        return self._initUrl
    
    @classmethod
    def setContent(self, content):
        self._content = content
    
    @classmethod
    def getContent(self):
        return self._content

    @classmethod
    def fakeUserAgent(self, url):
        headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}

        dataTemp = Request(url, headers=headers)
        readData = urlopen(dataTemp).read()
        soup = BeautifulSoup(readData, "lxml")
        
        return soup
    
    