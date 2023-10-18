import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time
from datetime import datetime

import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
from flask import (
    Flask,
    redirect,
    url_for,
    render_template,
    request,
    flash,
    Response,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_, desc
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from data import BodyPart
from movenet import Movenet
import utils_pose as utils

db = SQLAlchemy()

app = Flask(__name__)

movenet = Movenet("movenet_thunder")
class_names = [
    "Chair",
    "Cobra",
    "Dolphin Plank",
    "Downward-Facing Dog",
    "Plank",
    "Side Plank",
    "Tree",
    "Warrior III",
    "Warrior II",
    "Warrior I",
]

list_dir = [
    "./avatars",
    "./models",
    "./uploaded",
]
for d in list_dir:
    if not os.path.exists(os.path.join("static", d)):
        os.makedirs(os.path.join("static", d))

label = ""
model_name = ""

app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploaded")
app.config["UPLOAD_MODEL"] = os.path.join("static", "models")
app.config["UPLOADED_PHOTOS_DEST"] = os.path.join("static", "avatars")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
app.config["SECRET_KEY"] = "mysecret"

db.init_app(app)


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
        self,
        modelname,
        location_h5,
        location_json,
        train_acc,
        val_acc,
        test_acc,
        selected="No",
    ):
        self.modelname = modelname
        self.location_h5 = location_h5
        self.location_json = location_json
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc
        self.selected = selected


with app.app_context():
    db.create_all()


def write_video(file_path, frames, fps):
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def resizew800(img_arr):
    original_height, original_width, _ = img_arr.shape
    scale = 800 / original_width
    resized_img = cv2.resize(img_arr, None, fx=scale, fy=scale)
    resized_height, resized_width, _ = resized_img.shape
    print(
        f"({original_width}, {original_height}) -> ({resized_width}, {resized_height})"
    )

    return resized_img


def find_newest_model_with_prefix(folder_path, prefix):
    file_list = os.listdir(folder_path)
    matching_files = [file for file in file_list if file.startswith(prefix)]

    if not matching_files:
        return None

    latest_file = max(
        matching_files, key=lambda x: os.path.getctime(os.path.join(folder_path, x))
    )
    # print(name_without_extension)
    name_without_extension = latest_file.rsplit(".", 1)[0]
    # print(latest_file)

    return name_without_extension


def load_model(model_name):
    # newest_model = find_newest_model_with_prefix("models", model_name)
    newest_model = find_newest_model_with_prefix("static/models", model_name)
    if model_name not in ["svm"]:
        # json_file = open("models/" + newest_model + ".json", "r")
        json_file = open("static/models/" + newest_model + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # model.load_weights("models/" + newest_model + ".h5")
        model.load_weights("static/models/" + newest_model + ".h5")
    else:
        # with open(f"models/{newest_model}.pkl", "rb") as model_file:
        with open(f"static/models/{newest_model}.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    return model


def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    return person


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )
    d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5

    return center


def normalize_pose_landmarks(landmarks):
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
        [
            [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints
        ],
        dtype=np.float32,
    )

    return pose_landmarks


def landmarks_to_embedding(landmarks_and_scores):
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


def get_skeleton(image):
    person = detect(image)
    pose_landmarks = get_keypoint_landmarks(person)
    lm_pose = landmarks_to_embedding(
        tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51))
    )

    return person, lm_pose


def predict_pose(model_name, model, lm_pose, class_names):
    global label
    if model_name not in ["svm"]:
        predict = model.predict(lm_pose)
    else:
        lm_pose = np.array(lm_pose)
        predict = model.predict_proba(lm_pose)
    label = class_names[np.argmax(predict)]
    acc_pred = np.max(predict[0], axis=0)
    rounded_acc_pred = round(acc_pred, 5)
    print("Class name: ", label)
    print("Acurracy: ", acc_pred)

    label = f"{label} | {str(rounded_acc_pred)}"

    return label


def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True, keep_input_size=False
):
    image_np = utils.visualize(image, [person])
    # height, width, channel = image.shape
    # aspect_ratio = float(width) / height
    # fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # im = ax.imshow(image_np)
    # if close_figure:
    #     plt.close(fig)
    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np


def draw_class_name_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 50)
    fontScale = 1
    # color = (13, 110, 253)
    # color = (253, 110, 13)
    color = (19, 255, 30)
    thickness = 2
    lineType = 2
    img = cv2.putText(img, label, org, font, fontScale, color, thickness, lineType)

    return img


def result_pre(list_label):
    label_pose = []
    counts = Counter(item.split(" | ")[0] for item in list_label)
    most_common_value = counts.most_common(1)[0]
    label_pre = most_common_value[0]
    n_label_pre = most_common_value[1]
    print(f"{label_pre} : {n_label_pre}")

    return label_pre


def generate_webcam(class_names):
    global model, model_name
    cap = cv2.VideoCapture(0)
    i = 0
    label = ""
    n_frames = 5
    frame_buffer = []
    executor = ThreadPoolExecutor(max_workers=1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tf.convert_to_tensor(img, dtype=tf.uint8)
        frame_buffer.append(img)
        # i = i + 1
        # print(f"Start detect: frame {i}")
        person, lm_pose = get_skeleton(img)
        img = np.array(img)
        img = draw_prediction_on_image(
            img, person, crop_region=None, close_figure=False, keep_input_size=True
        )
        if len(frame_buffer) == n_frames:
            future = executor.submit(
                lambda: predict_pose(model_name, model, lm_pose, class_names)
            )
            label = future.result()

            frame_buffer.clear()

        img = draw_class_name_on_image(label, img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode(".jpg", img)
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

        cv2.destroyAllWindows()
    cap.release()


def generate_video(path):
    global model, model_name
    cap = cv2.VideoCapture(path)
    i = 0
    label = ""
    list = []
    list_label = []
    n_frames = 5
    frame_buffer = []
    scale = 0.5
    executor = ThreadPoolExecutor(max_workers=1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tf.convert_to_tensor(img, dtype=tf.uint8)
        frame_buffer.append(img)
        # i = i + 1
        # print(f"Start detect: frame {i}")
        person, lm_pose = get_skeleton(img)
        img = np.array(img)
        img = draw_prediction_on_image(
            img, person, crop_region=None, close_figure=False, keep_input_size=True
        )
        if len(frame_buffer) == n_frames:
            future = executor.submit(
                lambda: predict_pose(model_name, model, lm_pose, class_names)
            )
            label = future.result()
            list_label.append(label)

            frame_buffer.clear()

        img = draw_class_name_on_image(label, img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        list.append(img)
        cv2.destroyAllWindows()
    cap.release()
    write_video(path, np.array(list), 30)

    return result_pre(list_label)


def generate_image(path):
    global model, model_name
    executor = ThreadPoolExecutor(max_workers=1)
    img_arr = cv2.imread(path)
    if img_arr.shape[1] > 800:
        with tempfile.TemporaryDirectory() as temp_dir:
            img_resized = resizew800(img_arr)
            temp_image_path = os.path.join(temp_dir, "resized_image.jpg")
            cv2.imwrite(temp_image_path, img_resized)

            image = tf.io.read_file(temp_image_path)
            img = tf.image.decode_image(image)
    else:
        image = tf.io.read_file(path)
        img = tf.image.decode_image(image)

    person, lm_pose = get_skeleton(img)
    img = np.array(img)
    img = draw_prediction_on_image(
        img, person, crop_region=None, close_figure=False, keep_input_size=True
    )
    future_img = executor.submit(
        lambda: predict_pose(model_name, model, lm_pose, class_names)
    )
    label = future_img.result()

    img = draw_class_name_on_image(label, img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    result = label

    return result


def stream_video(path):
    cam = cv2.VideoCapture(path)
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            time.sleep(0.01)
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html", class_names=class_names)


@app.route("/dashboard")
def index_admin():
    user_count = db.session.query(User).filter(User.role == "User").count()
    model_count = db.session.query(Models).count()
    upload_count = db.session.query(Upload).count()
    return render_template(
        "admin_home.html",
        user_count=user_count,
        model_count=model_count,
        upload_count=upload_count,
        class_names=class_names,
    )


@app.route("/upload")
def upload():
    global model_name, model, mod_id
    if session.get("user_id") is None:
        return redirect(url_for("login"))
    else:
        has_mod = db.session.query(Models).first() is not None
        if has_mod:
            mod = Models.query.filter(Models.selected == "Yes").first()
            if mod:
                model_name = mod.modelname
                mod_id = mod.id
                model = load_model(model_name)
            else:
                flash("Please select model", "warning")
                mod = Models.query.all()
                return render_template("admin_model.html", listModel=mod)

        return render_template("upload.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    global tmp_path, mod_id

    if request.method == "POST":
        if not request.files["file"]:
            flash("Please choose your video or image", "warning")
            return redirect(url_for("upload"))
        else:
            file = request.files["file"]
            filename = file.filename
            image_extensions = {"jpg", "jpeg", "png", "gif"}
            video_extensions = {"mp4", "mov", "avi"}

            f_name, file_extension = filename.split(".")
            t = datetime.now()
            t = t.strftime("%H%M%S")

            file_name = f"{f_name}_{t}.{file_extension}"
            print(file_name)
            path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
            print(path)
            file.save(path)

            tmp_path = path

            if file_extension in video_extensions:
                result_pre = generate_video(path)
            elif file_extension in image_extensions:
                result_pre = generate_image(path)
            else:
                flash("Please choose your video or image", "error")
                return redirect(url_for("upload"))

            now = datetime.now()
            time = now.strftime("%d-%m-%Y %H-%M-%S")
            uploaded_file = Upload(
                file_name, time, result_pre, session["user_id"], mod_id
            )
            db.session.add(uploaded_file)
            db.session.commit()

            return redirect(url_for("play_upload", upload_id=uploaded_file.id))

    else:
        return render_template("index.html")


@app.route("/play_upload/<int:upload_id>", methods=["GET", "POST"])
def play_upload(upload_id):
    global tmp_path
    file = Upload.query.filter_by(id=upload_id).first()
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], file.location)
    return render_template("show_result.html")


@app.route("/stream")
def stream():
    global tmp_path
    return Response(
        stream_video(tmp_path), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/delete_upload/<int:upload_id>", methods=["POST"])
def delete_upload(upload_id):
    upload = Upload.query.get(upload_id)
    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], upload.location))
    db.session.delete(upload)
    db.session.commit()

    if session["role"] in ["Admin"]:
        upload = Upload.query.all()
        return render_template("admin_history.html", listUpload=upload)
    else:
        upload = (
            Upload.query.filter_by(user_id=session["user_id"])
            .order_by(desc(Upload.time))
            .all()
        )
        return render_template("history.html", listUpload=upload)


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        generate_webcam(class_names),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/history")
def history():
    if session.get("user_id") is None:
        return redirect(url_for("login"))
    else:
        if session["role"] in ["Admin"]:
            uploaded = Upload.query.order_by(desc(Upload.time)).all()
            return render_template("admin_history.html", listUpload=uploaded)

        else:
            uploaded = (
                Upload.query.filter_by(user_id=session["user_id"])
                .order_by(desc(Upload.time))
                .all()
            )
            return render_template("history.html", listUpload=uploaded)


@app.route("/admin/add_model")
def add_model():
    return render_template("admin_add_model.html")


@app.route("/admin/add_model", methods=["GET", "POST"])
def admin_add_model():
    if session["role"] in ["Admin"]:
        if request.method == "POST":
            t = datetime.now()
            t = t.strftime("%H%M%S")

            f_h5 = request.files["file_h5"]
            file_h5 = f_h5.filename
            f_name_h5, f_extension_h5 = file_h5.rsplit(".", 1)
            file_name_h5 = f"{f_name_h5}_{t}.{f_extension_h5}"
            path_h5 = os.path.join(app.config["UPLOAD_MODEL"], file_name_h5)
            f_h5.save(path_h5)

            f_json = request.files["file_json"]
            file_json = f_json.filename
            f_name_json, f_extension_json = file_json.rsplit(".", 1)
            file_name_json = f"{f_name_json}_{t}.{f_extension_json}"
            path_json = os.path.join(app.config["UPLOAD_MODEL"], file_name_json)
            f_json.save(path_json)

            mod = Models(
                request.form["modelname"],
                path_h5,
                path_json,
                request.form["train_acc"],
                request.form["val_acc"],
                request.form["test_acc"],
            )

            db.session.add(mod)
            db.session.commit()

        mod = Models.query.all()
        return render_template("admin_model.html", listModel=mod)


@app.route("/admin/delete_model/<int:model_id>", methods=["POST"])
def delete_model(model_id):
    if session["role"] in ["Admin"]:
        mod = Models.query.get(model_id)
        os.remove(mod.location_h5)
        os.remove(mod.location_json)
        db.session.delete(mod)
        db.session.commit()

        mod = Models.query.all()
        return render_template("admin_model.html", listModel=mod)


@app.route("/model")
def model():
    mod = Models.query.all()
    return render_template("admin_model.html", listModel=mod)


@app.route("/select_model/<int:model_id>", methods=["GET", "POST"])
def select_model(model_id):
    global model_name, model, mod_id
    mod_id = model_id
    selected_models = Models.query.filter(Models.selected == "Yes").first()
    if selected_models:
        selected_models.selected = "No"
        db.session.commit()

    mod = Models.query.get(model_id)
    mod.selected = "Yes"
    model_name = mod.modelname
    db.session.commit()

    model = load_model(model_name)

    mod = Models.query.all()
    return render_template("admin_model.html", listModel=mod)


@app.route("/admin/user")
def user():
    if session["role"] in ["Admin"]:
        user = User.query.all()
        return render_template("admin_user.html", listUser=user)


@app.route("/admin/add_user")
def add_user():
    if session["role"] in ["Admin"]:
        return render_template("admin_add_user.html")


@app.route("/admin/add_user", methods=["GET", "POST"])
def admin_add_user():
    if session["role"] in ["Admin"]:
        if request.method == "POST":
            hashed_password = generate_password_hash(
                request.form["password"], method="pbkdf2:sha256"
            )

            file = request.files["avatar"]
            # filename = file.filename
            file_name = f"{str(int(datetime.now().timestamp()))}.jpg"

            file_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], file_name)
            file.save(file_path)

            user = User(
                request.form["username"],
                request.form["email"],
                hashed_password,
                file_name,
            )

            db.session.add(user)
            db.session.commit()

        user = User.query.all()
        return render_template("admin_user.html", listUser=user)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if session["role"] in ["Admin"]:
        user = User.query.get(user_id)
        uploads = Upload.query.filter_by(user_id=user.id).all()
        for upload in uploads:
            db.session.delete(upload)
            os.remove(upload.location)

        avt_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], user.avatar)
        os.remove(avt_path)
        db.session.delete(user)
        db.session.commit()
        user = User.query.all()
        return render_template("admin_user.html", listUser=user)


@app.route("/admin/<int:user_id>", methods=["GET", "POST"])
def update_user(user_id):
    if session["role"] in ["Admin"]:
        user = User.query.get(user_id)
        return render_template("admin_update_user.html", user=user)


@app.route("/admin/update_user/<int:user_id>", methods=["GET", "POST"])
def admin_update_user(user_id):
    if session["role"] in ["Admin"]:
        user = User.query.get(user_id)
        if request.method == "POST":
            if "password" in request.form and request.form["password"]:
                hashed_password = generate_password_hash(
                    request.form["password"], method="pbkdf2:sha256"
                )
                user.password = hashed_password

            if "avatar" in request.files:
                file = request.files["avatar"]
                if file:
                    filename = file.filename
                    file_path = os.path.join(
                        app.config["UPLOADED_PHOTOS_DEST"], filename
                    )
                    user.avatar = file_path

            if "username" in request.form:
                user.username = request.form["username"]

            if "email" in request.form:
                user.email = request.form["email"]

            db.session.commit()

            user = User.query.all()
            return render_template("admin_user.html", listUser=user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email_username = request.form.get("email_username")
        password = request.form.get("password")
        remember = True if request.form.get("remember") else False

        user = User.query.filter(
            or_(User.username == email_username, User.email == email_username)
        ).first()

        if user:
            if check_password_hash(user.password, password):
                session["user_id"] = user.id
                session["user_name"] = user.username
                # session["avatar"] = os.path.join('avatars', user.avatar)
                session["avatar"] = f"avatars/{user.avatar}"
                print(session["avatar"])
                session["role"] = user.role

                if session["role"] in ["Admin"]:
                    return redirect(url_for("index_admin"))
                else:
                    return redirect(url_for("index"))

            else:
                flash("Enter incorrect password", "warning")
                return render_template("login.html")
        else:
            flash("User does not exist", "warning")
            return render_template("login.html")
    else:
        return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        hashed_password = generate_password_hash(
            request.form["password"], method="pbkdf2:sha256"
        )

        file = request.files["avatar"]
        file_name = f"{str(int(datetime.now().timestamp()))}.jpg"
        file_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], file_name)
        file.save(file_path)

        user = User(
            request.form["username"], request.form["email"], hashed_password, file_name
        )

        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    else:
        return render_template("register.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("avatar", None)
    session.pop("role", None)

    return redirect(url_for("index"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, threaded=True)
