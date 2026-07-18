from flask import Flask, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('webapp', 'index.html')

@app.route('/<path:path>')
def files(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')