from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

app = Flask(__name__)

# Cấu hình cơ sở dữ liệu SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.config["SECRET_KEY"] = "mysecret"

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    avatar = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="User")
    uploads = db.relationship("Upload", back_populates="user")

    def __init__(self, username, email, password, avatar, role="User"):
        self.username = username
        self.email = email
        self.password = password
        self.avatar = avatar
        self.role = role


class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100), unique=True, nullable=False)
    time = db.Column(db.String(50), unique=True, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("models.id"), nullable=False)
    user = db.relationship("User", back_populates="uploads")
    model = db.relationship("Models", back_populates="upload")

    def __init__(self, location, time, result, user_id, model_id):
        self.location = location
        self.time = time
        self.result = result
        self.user_id = user_id
        self.model_id = model_id


class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    modelname = db.Column(db.String(100), unique=True, nullable=False)
    location_h5 = db.Column(db.String(100), unique=True, nullable=False)
    location_json = db.Column(db.String(100), unique=True, nullable=False)
    train_acc = db.Column(db.Float, nullable=False)
    val_acc = db.Column(db.Float, nullable=False)
    test_acc = db.Column(db.Float, nullable=False)
    selected = db.Column(db.String(10), nullable=False, default="No")
    upload = db.relationship("Upload", back_populates="model")

    def __init__(
        self, modelname, location_h5, location_json, train_acc, val_acc, test_acc, selected="No"
    ):
        self.modelname = modelname
        self.location_h5 = location_h5
        self.location_json = location_json
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc
        self.selected = selected

# Khởi tạo Flask-Admin
admin = Admin(app, name='Admin Panel', template_mode='bootstrap4')

# Định nghĩa lớp ModelView cho mô hình người dùng
class UserView(ModelView):
    column_list = ['username', 'role', 'email', 'password', 'avatar', 'uploads']
    column_searchable_list = ['username', 'email']  # Cho phép tìm kiếm theo tên người dùng hoặc email
    column_filters = ['role']
    column_formatters = {
        'password': lambda v, c, m, p: m.password[:25] + " . . ." if len(m.password) > 25 else m.password,
        'avatar': lambda v, c, m, p: m.avatar[:25] + " . . ." if len(m.avatar) > 25 else m.avatar,
        'uploads': lambda v, c, m, p: len(m.uploads)
    }
    column_labels = {
        'uploads': 'File'
    }

class UploadView(ModelView):
    column_list = ['user.username', 'location', 'time','model.modelname','result']
    column_labels = {
        'user.username': 'Username',
        'location': 'File name',
        'time': 'Upload time',
        'model.modelname': 'Model',
        'result': 'Result'
    }
    column_searchable_list = ['result']
    column_filters = ['user.username']


class ModelsView(ModelView):
    column_list = ['modelname', 'train_acc', 'val_acc', 'test_acc', 'selected', 'location_h5', 'location_json']
    column_formatters = {
        'location_h5': lambda v, c, m, p: m.location_h5[:25] + " . . ." if len(m.location_h5) > 25 else m.location_h5,
        'location_json': lambda v, c, m, p: m.location_json[:25] + " . . ." if len(m.location_json) > 25 else m.location_json,
    }
    
# Thêm mô hình người dùng vào giao diện quản trị
admin.add_view(UserView(User, db.session))
admin.add_view(UploadView(Upload, db.session))
admin.add_view(ModelsView(Models, db.session))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(port=8080, debug=True, threaded=True)
