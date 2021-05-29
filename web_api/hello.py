
"""
Hello API route handlers
"""
from flask import jsonify

from . import api


@api.route('/hello/<name>')
def hello(name):
    return jsonify(dict(hello=name))