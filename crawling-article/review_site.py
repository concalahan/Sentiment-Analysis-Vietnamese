from review_site_interface import reviewSite
from file_process import *
from constants import *

class Tinhte(reviewSite):
    
    def __init__(self):
        pass
    
    def getArticle(self, initUrl):
        self.setInitUrl(initUrl)

        try:
            soup = self.fakeUserAgent(self.getInitUrl())
        except Exception as e:
            raise e

        articlesTag = soup.find("article")

        content = articlesTag.get_text()
        content = content.strip()
        content = content.replace("\n", "")

        return content
    
    def storeContent(self):
        # EX : https://tinhte.vn/threads/review-chi-tiet-iphone-xs-max-apple-da-day-hu-nguoi-dung-nhu-the-nao.2858501/
        url = self.getInitUrl()

        # Get the left site of the url
        fileName = (url.split("."))[0]
        
        # Get the last string after split by "/"
        fileName = (fileName.split("/"))[-1]

        createDirectory(DATA_DIRECTORY)

        path = DATA_DIRECTORY + "/" + fileName + ".txt"

        writeTextFile(path, self.getContent())

class Vnreview(reviewSite):

    def __init__(self):
        pass
    
    def getArticle(self, initUrl):
        self.setInitUrl(initUrl)

        try:
            soup = self.fakeUserAgent(self.getInitUrl())
        except Exception as e:
            raise e

        articlesTag = soup.find("div", {"class": "journal-content-article"})

        content = articlesTag.get_text()

        self.setContent(content)

        return content
    
    def storeContent(self):
        # EX : https://vnreview.vn/danh-gia-chi-tiet-di-dong/-/view_content/content/2655916/danh-gia-iphone-xs-max-lon-nhat-dat-nhat-tot-nhat
        url = self.getInitUrl()
        
        # Get the last string after split by "/"
        fileName = (url.split("/"))[-1]

        createDirectory(DATA_DIRECTORY)

        path = DATA_DIRECTORY + "/" + fileName + ".txt"

        writeTextFile(path, self.getContent())


if __name__ == "__main__":
    tinhte = Tinhte()
    # print(tinhte.getContent("https://tinhte.vn/threads/review-chi-tiet-iphone-xs-max-apple-da-day-hu-nguoi-dung-nhu-the-nao.2858501/"))
    
    # print(tinhte.getContent())

    vnreview = Vnreview()
    print(vnreview.getArticle("https://vnreview.vn/danh-gia-chi-tiet-di-dong/-/view_content/content/2655916/danh-gia-iphone-xs-max-lon-nhat-dat-nhat-tot-nhat"))
        