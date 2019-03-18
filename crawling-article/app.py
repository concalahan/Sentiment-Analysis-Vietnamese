from review_site import *

from flask import Flask
from flask import request
from flask import abort, redirect, url_for
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/get-article', methods=['GET', 'POST'])
def getContentReviewSite():
    if(request.method == 'POST'):
        # Call object
        tinhte = Tinhte()
        vnreview = Vnreview()

        # Get json 
        url = request.get_json().get("q")
        content = ""
        
        if "tinhte" in url:
            content = tinhte.getArticle(url)
        elif "vnreview" in url:
            content = vnreview.getArticle(url)
            
        return content
    else:
        abort(400)
        return 'ONLY ACCEPT POST REQUEST'
    