from flask import Flask, render_template, request
from flask_cors import CORS

from web_api import api as api_blueprint
from errors import add_error_handlers

from image_morphing import get_random_morph
from utils import serve_pil_image


def create_app():
    app = Flask(__name__, static_url_path='', 
        static_folder='web/static', template_folder='web/templates'
    )
    CORS(app, resources={r'/*': {'origins': '*'}})
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')
    add_error_handlers(app)
    return app

app = create_app()
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.cache = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/new_morph", methods=["GET", "POST"])
def new_random_morph_request():
    current_name_1, current_name_2 = None, None
    img, current_name_1, current_name_2 = get_random_morph()
    app.cache["name1"] = current_name_1
    app.cache["name2"] = current_name_2
    return serve_pil_image(img, "png"), 200

@app.route("/answer", methods=["GET", "POST"])
def get_answer():
    res = {"name1": app.cache["name1"], "name2": app.cache["name2"]}
    app.cache["name1"] = None
    app.cache["name2"] = None
    return res, 200


# Disable cache
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)