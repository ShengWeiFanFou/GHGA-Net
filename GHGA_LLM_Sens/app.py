import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
from extractor.fileTypeVar import get_file_type
from extractor.Extractor import deZipWord,getAllFilesWord
from predict import data_for_predict

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['txt', 'docx', 'pptx', 'xlsx', 'cebx','pdf'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    print(datetime.datetime.now(), file.filename)
    ext=file.filename.rsplit('.', 1)[1]
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(src_path)
    type = get_file_type(src_path)
    if type!=None:
        ext = type.extension
        if ext in ALLOWED_EXTENSIONS:
            filename = file.filename.split('.')[0] + '.' + ext
            return jsonify({'status': 1,
                            'image_url': 'http://127.0.0.1:5003/img/{}.png'.format(ext),
                            'draw_url': 'http://127.0.0.1:5003/img/txt.png',
                            'image_info': ['case2.docx', 'TopSecret', 'ERNIE3-Sens', 'TopSecret:90.17%, Secret:9.83%'],
                            'nowfile': os.path.join(app.config['UPLOAD_FOLDER'], filename)}
                           )
    return jsonify({'status': 0})


@app.route('/extract', methods=['GET', 'POST'])
def extract_file():
    file = request.get_json()['upfile']
    filename=file.split('\\')[1].split('.')[0]
    print(filename)
    path="org_file_upload/file_"+filename
    os.makedirs(path)
    shutil.copy(file, path)
    # 先以word为例
    deZipWord(path)
    getAllFilesWord(path)
    current_path="{}/out/{}/text/{}.txt".format(path,filename,filename)
    return jsonify({'status': 1,
                        'draw_url': 'http://127.0.0.1:5003/img/txt.png',
                        'image_info': ['case2.docx','TopSecret','ERNIE3-Sens','TopSecret:90.17%, Secret:9.83%'],
                        'nowfile':current_path,
                    'filename':filename+'.txt'})


@app.route('/predict', methods=['GET', 'POST'])
def predictfile():
    file = request.get_json()['upfile']
    label,llist,time,flag=data_for_predict(file)

    return jsonify({'status': 1,'label': label,
                        'list': llist,
                    'time':time,'flag':flag})


@app.route('/download_txt', methods=['GET', 'POST'])
def download_txt_file():
    file = request.get_json()['upfile']
    return send_file(file, as_attachment=True)


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


# show photo
@app.route('/img/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'img/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    files = ['uploads', 'org_file_upload' ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    app.run(host='127.0.0.1', port=5003, debug=True)
