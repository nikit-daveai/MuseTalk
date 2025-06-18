
import random
import csv
import os
from flask import Flask, request, render_template_string

app = Flask(__name__)
VIDEO_LIST = []
RESULT_CSV = 'mos_results.csv'

MOS_TEMPLATE = """
<html>
<head><title>MOS Evaluation</title></head>
<body>
<h2>Rate the realism of each video from 1 (poor) to 5 (excellent):</h2>
<form action="/submit" method="post">
  {% for idx, src in videos %}
  <div>
    <video width="480" controls>
      <source src="{{ src }}" type="video/mp4">
    </video><br>
    <label>Rating (1-5):</label>
    <input type="number" name="vid{{ idx }}" min="1" max="5" required>
  </div><br>
  {% endfor %}
  <input type="submit" value="Submit">
</form>
</body>
</html>
"""

@app.route('/')
def form():
    return render_template_string(MOS_TEMPLATE, videos=list(enumerate(VIDEO_LIST)))

@app.route('/submit', methods=['POST'])
def submit():
    ratings = request.form
    with open(RESULT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [ratings.get(f'vid{i}', '') for i in range(len(VIDEO_LIST))]
        writer.writerow(row)
    return '<h3>Thank you! Your ratings have been recorded.</h3>'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', nargs='+', required=True, help='List of video files')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--csv', default='mos_results.csv')
    args = parser.parse_args()

    VIDEO_LIST = args.videos
    RESULT_CSV = args.csv

    print(f'Open http://localhost:{args.port} in a browser to start rating.')
    app.run(host='0.0.0.0', port=args.port)
