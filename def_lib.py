import os
import csv
import itertools
import tempfile
import datetime
import pickle

import tqdm
import psutil
import wget
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
from matplotlib import pyplot as plt
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data import BodyPart
from movenet import Movenet
import utils_pose as utils

import truongmodel

"""
=================================================== DEF MAKE FILE CSV ===================================================
"""

if "movenet_thunder.tflite" not in os.listdir():
    wget.download(
        "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
        "movenet_thunder.tflite",
    )


# Load MoveNet Thunder model
movenet = Movenet("movenet_thunder")


def detect(input_tensor, inference_count=3):
    # Detect pose using the full input image
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor.numpy(), reset_crop_region=False)

    return person


class MoveNetPreprocessor(object):
    #     this class preprocess pose samples, it predicts keypoints on the images
    #     and save those keypoints in a csv file for the later use in the classification task
    def __init__(self, images_in_folder, csvs_out_path):
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._message = []
        #       Create a temp dir to store the pose CSVs per class
        self._csvs_out_folder_per_class = tempfile.mkdtemp()
        self._pose_class_names = sorted([n for n in os.listdir(images_in_folder)])

    def process(self, detection_threshold=0.1):
        # Preprocess the images in the given folder
        for pose_class_name in self._pose_class_names:
            # Paths for pose class
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            csv_out_path = os.path.join(
                self._csvs_out_folder_per_class, pose_class_name + ".csv"
            )

            # Detect landmarks in each image and write it to the csv files
            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )

                # Get the list of images
                image_names = sorted([n for n in os.listdir(images_in_folder)])

                valid_image_count = 0

                # Detect pose landmarks in each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)

                    try:
                        image = tf.io.read_file(image_path)
                        image = tf.io.decode_jpeg(image)
                    except:
                        self._message.append("Skipped" + image_path + " Invalid image")
                        continue

                    # Skip images that is not RGB
                    if image.shape[2] != 3:
                        self._message.append(
                            "Skipped" + image_path + " Image is not in RGB"
                        )
                        continue

                    person = detect(image)

                    # Save landmarks if all landmarks above than the threshold
                    min_landmark_score = min(
                        [keypoint.score for keypoint in person.keypoints]
                    )
                    should_keep_image = min_landmark_score >= detection_threshold
                    if not should_keep_image:
                        self._message.append(
                            "Skipped"
                            + image_path
                            + ". No pose was confidentlly detected."
                        )
                        continue

                    valid_image_count += 1

                    # Get landmarks and scale it to the same size as the input image
                    pose_landmarks = np.array(
                        [
                            [
                                keypoint.coordinate.x,
                                keypoint.coordinate.y,
                                keypoint.score,
                            ]
                            for keypoint in person.keypoints
                        ],
                        dtype=np.float32,
                    )

                    # writing the landmark coordinates (tọa độ) to its csv files
                    # coord = pose_landmarks.flatten().astype(np.str).tolist()
                    coord = pose_landmarks.flatten().astype(np.str_).tolist()
                    csv_out_writer.writerow([image_name] + coord)

        # Print the error message collected during preprocessing.
        print(self._message)

        # Combine all per-csv class CSVs into a sigle csv file
        all_landmarks_df = self.all_landmarks_as_dataframe()
        all_landmarks_df.to_csv(self._csvs_out_path, index=False)

    def class_names(self):
        """List of classes found in the training dataset."""
        return self._pose_class_names

    def all_landmarks_as_dataframe(self):
        # Merging all csv for each class into a single csv file
        total_df = None
        for class_index, class_name in enumerate(self._pose_class_names):
            csv_out_path = os.path.join(
                self._csvs_out_folder_per_class, class_name + ".csv"
            )
            per_class_df = pd.read_csv(csv_out_path, header=None)

            # Add the labels
            per_class_df["class_no"] = [class_index] * len(per_class_df)
            per_class_df["class_name"] = [class_name] * len(per_class_df)

            # Append the folder name to the filename first column
            per_class_df[per_class_df.columns[0]] = (
                class_name + "/" + per_class_df[per_class_df.columns[0]]
            )

            if total_df is None:
                # For the first class, assign(gán) its data to the total dataframe
                total_df = per_class_df
            else:
                # Concatenate(nối) each class's data into the total dataframe
                total_df = pd.concat([total_df, per_class_df], axis=0)

        list_name = [
            [bodypart.name + "_x", bodypart.name + "_y", bodypart.name + "_score"]
            for bodypart in BodyPart
        ]

        header_name = []
        for columns_name in list_name:
            header_name += columns_name
        header_name = ["file_name"] + header_name
        header_map = {
            total_df.columns[i]: header_name[i] for i in range(len(header_name))
        }

        total_df.rename(header_map, axis=1, inplace=True)

        return total_df


"""
======================================================= DEF TRAIN =======================================================
"""


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Drop the file_name columns as you don't need it during training.
    df.drop(["file_name"], axis=1, inplace=True)
    # Extract the list of class names
    classes = df.pop("class_name").unique()
    # Extract the labels
    y = df.pop("class_no")
    # Convert the input features and labels into the correct format for training.
    X = df.astype("float64")
    y = keras.utils.to_categorical(y)

    return X, y, classes


def load_csv_svm(csv_path):
    df = pd.read_csv(csv_path)
    # Drop the file_name columns as you don't need it during training.
    df.drop(["file_name"], axis=1, inplace=True)
    # Extract the list of class names
    classes = df.pop("class_name").unique()
    # Extract the labels
    y = df.pop("class_no")
    # Convert the input features and labels into the correct format for training.
    X = df.astype("float64")
    # y should be in array form compulsory.
    y = np.array(y)

    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5

    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(
            tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51))
        )
        processed_X_train.append(tf.reshape(embedding, (34)))

    return tf.convert_to_tensor(processed_X_train)


"""
===================================================== DEF SAVE FILE =====================================================
"""


def save_model_summary(folder_saved, name_saved, model):
    model_name = name_saved.split("_")[0]
    model_summary_path = f"{folder_saved}/summary_{name_saved}.txt"
    if model_name == "svm":
        data = f"C: {model.C}, Kernel: {model.kernel}, Gamma: {model.gamma}, Degree: {model.degree}, Decision function: {model.decision_function_shape}"
        with open(model_summary_path, "w") as f:
            f.write(data)

    else:
        # Mở tệp và sử dụng redirection để ghi dữ liệu kiến trúc vào tệp
        with open(model_summary_path, "w") as f:
            with redirect_stdout(f):
                model.summary()


def save_log_txt(folder_saved, name_saved, history_callback):
    ep_arr = range(1, len(history_callback.history["accuracy"]) + 1, 1)
    train_acc = history_callback.history["accuracy"]
    val_acc = history_callback.history["val_accuracy"]
    train_loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    title_cols = np.array(
        ["epoch", "train_acc", "valid_acc", "train_loss", "valid_loss"]
    )
    res = (ep_arr, train_acc, val_acc, train_loss, val_loss)
    res = np.transpose(res)
    combined_res = np.array(np.vstack((title_cols, res)))
    np.savetxt(
        f"{folder_saved}/{name_saved}.txt", combined_res, fmt="%s", delimiter="\t"
    )


def save_log_csv(folder_saved, name_saved, history_callback):
    hist_df = pd.DataFrame(history_callback.history)
    hist_df.index += 1
    hist_df.to_csv(f"{folder_saved}/{name_saved}.csv", index_label="epoch")


def save_measure(
    folder_saved,
    name_saved,
    samples_training,
    samples_validation,
    samples_test,
    elapsed,
    consumes_memory,
    history_callback,
):
    num_epoch = len(history_callback.history["accuracy"])
    best_train_accuracy = max(history_callback.history["accuracy"])
    best_train_epoch = (
        history_callback.history["accuracy"].index(best_train_accuracy) + 1
    )
    best_val_accuracy = max(history_callback.history["val_accuracy"])
    best_val_epoch = (
        history_callback.history["val_accuracy"].index(best_val_accuracy) + 1
    )
    model_training = "Samples training set: " + str(
        samples_training
    ) + "\n" + "Samples validation set: " + str(
        samples_validation
    ) + "\n" + "Samples test set: " + str(
        samples_test
    ) + "\n" "Running time: " + str(
        elapsed
    ) + "\n" + "Memory consumed during training (%): " + str(
        consumes_memory
    ) + "\n" + "Total epochs: " + str(
        num_epoch
    ) + "\n" + "Best train accuracy: " + str(
        best_train_accuracy
    ) + " / epoch: " + str(
        best_train_epoch
    ) + "\n" + "Best val accuracy: " + str(
        best_val_accuracy
    ) + " / epoch: " + str(
        best_val_epoch
    )
    print(model_training)
    with open(f"{folder_saved}/measure_{name_saved}.txt", "w") as f:
        f.writelines(model_training)
    f.close()


def save_evaluate(folder_saved, name_saved, model, X_train, y_train, X_test, y_test):
    loss_train, accuracy_train = model.evaluate(X_train, y_train)
    loss_test, accuracy_test = model.evaluate(X_test, y_test)
    data_eval = (
        "LOSS TRAIN: "
        + str(loss_train)
        + " / ACCURACY TRAIN: "
        + str(accuracy_train)
        + "\n"
        + "LOSS TEST: "
        + str(loss_test)
        + " / ACCURACY TEST: "
        + str(accuracy_test)
    )
    print(data_eval)

    with open(f"{folder_saved}/evaluation_{name_saved}.txt", "w") as file:
        file.writelines(data_eval)
    file.close()


def plot_acc(folder_saved, name_saved, history_callback):
    plt.plot(history_callback.history["accuracy"])
    plt.plot(history_callback.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(f"{folder_saved}/accuracy_{name_saved}.png")
    plt.close()


def plot_loss(folder_saved, name_saved, history_callback):
    plt.plot(history_callback.history["loss"])
    plt.plot(history_callback.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(f"{folder_saved}/loss_{name_saved}.png")
    plt.close()


def plot_confusion_matrix(
    plot_confusion_matrix_path,
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """Plots the confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig = plt.gcf()
    fig.set_size_inches(12.5, 10.5)  # Đặt kích thước hình ảnh
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
    fig.savefig(plot_confusion_matrix_path)  # Lưu hình ảnh
    plt.close()


def plot_bar_chart_svm(folder_saved, name_saved, title, classes, data, ylabel):
    colors = [
        "#EF9595",
        "#EFB495",
        "#EFD595",
        "#EBEF95",
        "#94A684",
        "#AEC3AE",
        "#E4E4D0",
        "#FFB6D9",
        "#FCBAAD",
        "#FFDBAA",
    ]
    # Vẽ biểu đồ cột
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(classes, data, color=colors)
    plt.xlabel("Nhãn ")
    plt.ylabel(f"{ylabel} (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Hiển thị giá trị Recall trên từng cột
    for i, recall in enumerate(data):
        plt.annotate(
            f"{recall:.5f}",
            (i, recall),
            ha="center",
            va="center",
            textcoords="offset points",
            xytext=(0, 5),
        )
    plt.savefig(f"{folder_saved}/{name_saved}.png")
    plt.close()


def k_fold_validation_svm(folder_saved, name_saved, model, X_train, y_train, k):
    scores = cross_val_score(model, X_train, y_train, cv=k, scoring="accuracy")
    # In kết quả accuracy của từng fold
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: Accuracy = {score:.5f}")
    # In tổng hợp kết quả cross-validation
    print(f"Accuracy trung bình trên {k} folds: {scores.mean():.5f}")
    print("=" * 100)
    # Tạo một mảng numpy chứa cả giá trị scores.mean()
    results = np.append(scores, scores.mean())
    # Lưu kết quả vào file txt
    np.savetxt(
        f"{folder_saved}/cross_validation_{k}_fold_{name_saved}.txt",
        results,
        fmt="%.5f",
    )


"""
========================================================= DEF EX ========================================================
"""


def load_data(path_data, test_size):
    # Đặt hạt ngẫu nhiên để đảm bảo khả năng tái huấn luyện
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_path = f"{path_data}/train_data.csv"
    test_path = f"{path_data}/test_data.csv"

    X, y, class_names = load_csv(train_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    # Load the test data
    X_test, y_test, _ = load_csv(test_path)

    print("Pre-process data...")
    processed_X_train = preprocess_data(X_train)
    processed_X_val = preprocess_data(X_val)
    processed_X_test = preprocess_data(X_test)

    return (
        processed_X_train,
        y_train,
        processed_X_val,
        y_val,
        processed_X_test,
        y_test,
        class_names,
    )


def load_svm_data(path_data):
    # Đặt hạt ngẫu nhiên để đảm bảo khả năng tái huấn luyện
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_path = f"{path_data}/train_data.csv"
    test_path = f"{path_data}/test_data.csv"
    # Load the train data
    X_train, y_train, class_names = load_csv_svm(train_path)
    X_test, y_test, _ = load_csv_svm(test_path)
    print("Pre-process data...")
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    processed_X_train = np.array(X_train)
    processed_X_test = np.array(X_test)

    return processed_X_train, y_train, processed_X_test, y_test, class_names


def run_exp(
    name_saved,
    model_name,
    processed_X_train,
    y_train,
    processed_X_val,
    y_val,
    processed_X_test,
    y_test,
    class_names,
    epochs,
    batch_size,
    e_patience,
    num_conv_layers_per_maxpool=2,
    num_maxpools=1,
    num_dense_layers=5,
):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    num_classes = len(class_names)

    samples_training = processed_X_train.shape[0]
    samples_validation = processed_X_val.shape[0]
    samples_test = processed_X_test.shape[0]

    ###################
    # Construct model #
    ###################
    if model_name in ["fcnn1d"]:
        name_saved = name_saved.replace("fcnn1d", "fcnn1d" + f"-{num_dense_layers}dense")
        model = truongmodel.fcnn1d_model(
            input_shape=(34),
            num_classes=num_classes,
            dropout_fc=0.2,
            num_dense_layers=num_dense_layers,  # 5
        )

    elif model_name in ["conv1d"]:
        name_saved = name_saved.replace(
            "conv1d",
            "conv1d"
            + f"-{num_maxpools}maxpool-{num_conv_layers_per_maxpool}convpermaxpool",
        )
        model = truongmodel.conv1d_model(
            input_shape=(34, 1),
            num_classes=num_classes,
            num_maxpools=num_maxpools,  # 1
            num_conv_layers_per_maxpool=num_conv_layers_per_maxpool,  # 2
            dropout_cnn=0.2,
            dropout_f=0.5,
        )

    # Lưu kiến trúc mô hình
    save_model_summary("model_training", name_saved, model)

    ####################
    # Network training #
    ####################
    checkpoint_path = f"save_models/weights.best_{name_saved}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    earlystopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=e_patience,
    )
    # Lấy thông tin bộ nhớ trước khi huấn luyện bắt đầu
    memory_info_before = psutil.virtual_memory()
    memory_usage_before = memory_info_before.percent
    start = datetime.datetime.now()
    history_callback = model.fit(
        processed_X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(processed_X_val, y_val),
        callbacks=[checkpoint, earlystopping],
    )
    # Lấy thông tin bộ nhớ sau khi huấn luyện kết thúc
    memory_info_after = psutil.virtual_memory()
    memory_usage_after = memory_info_after.percent
    elapsed = datetime.datetime.now() - start
    consumes_memory = memory_usage_after - memory_usage_before
    # memory_usage = round(consumes_memory, 5)
    print(f"Memory used before training: {memory_usage_before}%")
    print(f"Memory used after training: {memory_usage_after}%")
    print(f"Memory used: {consumes_memory} %")
    print("=" * 100)

    ####################
    #     Save log     #
    ####################
    save_log_txt("log", name_saved, history_callback)
    save_log_csv("log", name_saved, history_callback)
    ####################
    #     Save model   #
    ####################
    model.save(f"save_models/{name_saved}.h5")
    model_json = model.to_json()
    with open(f"save_models/{name_saved}.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    ####################
    #  model training  #
    ####################
    save_measure(
        "model_training",
        name_saved,
        samples_training,
        samples_validation,
        samples_test,
        elapsed,
        consumes_memory,
        history_callback,
    )

    ####################
    #  draw acc and loss  #
    ####################
    plot_acc("figures", name_saved, history_callback)
    plot_loss("figures", name_saved, history_callback)

    ####################
    #    EVALUATION    #
    ####################
    save_evaluate(
        "statistics",
        name_saved,
        model,
        processed_X_train,
        y_train,
        processed_X_test,
        y_test,
    )
    ####################
    # confusion matrix #
    ####################
    # Load model
    # model = tf.keras.models.load_model(f"save_models/{name_saved}.h5")

    y_pred = model.predict(processed_X_test)

    ytrue = np.argmax(y_test, axis=1)
    ypred = np.argmax(y_pred, axis=1)

    plot_confusion_matrix_path = f"figures/confusion_matrix_{name_saved}.png"
    plot_confusion_matrix_nor_path = f"figures/confusion_matrix_nor_{name_saved}.png"
    # Plot the confusion matrix
    cm = confusion_matrix(ytrue, ypred)
    plot_confusion_matrix(
        plot_confusion_matrix_path,
        cm,
        class_names,
        title="Confusion Matrix of Yoga Pose Model",
    )

    plot_confusion_matrix(
        plot_confusion_matrix_nor_path,
        cm,
        class_names,
        normalize=True,
        title="Normalized Confusion Matrix of Yoga Pose Model",
    )

    # Print the classification report
    print(
        "\nClassification Report:\n",
        classification_report(ytrue, ypred, target_names=class_names, zero_division=0),
    )

    with open(f"statistics/classification_report_{name_saved}.txt", "w") as f:
        f.writelines(
            classification_report(
                ytrue, ypred, target_names=class_names, zero_division=0
            )
        )
    f.close()


def run_svm_exp(
    name_saved,
    processed_X_train,
    y_train,
    processed_X_test,
    y_test,
    class_names,
    C,
    kernel,
    gamma,
    d,
    df,
    k_fold,
):
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    num_classes = len(class_names)

    ###################
    # Construct model #
    ###################
    model = truongmodel.svm_model(
        decision_function=df,
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=d,
        random_state=seed,
    )

    # Lưu kiến trúc mô hình
    save_model_summary("model_training", name_saved, model)

    ####################
    # Network training #
    ####################
    # Sử dụng cross_val_score để thực hiện cross-validation với 5 folds
    k_fold_validation_svm(
        "log", name_saved, model, processed_X_train, y_train, int(k_fold)
    )

    model.fit(processed_X_train, y_train)
    model_path = os.path.join(f"save_models/{name_saved}.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    # Tải mô hình từ tập tin
    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    # Sử dụng mô hình để dự đoán
    y_pred = loaded_model.predict(processed_X_test)

    # Đánh giá mô hình trên tập kiểm tra
    precision, recall, f1score, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    plot_bar_chart_svm(
        "figures",
        f"precision_{name_saved}",
        "Chỉ số Precision cho mỗi lớp",
        class_names,
        precision,
        "Precision",
    )

    plot_bar_chart_svm(
        "figures",
        f"recall_{name_saved}",
        "Chỉ số Recall cho mỗi lớp",
        class_names,
        recall,
        "Recall",
    )

    plot_bar_chart_svm(
        "figures",
        f"f1score_{name_saved}",
        "Chỉ số F score cho mỗi lớp",
        class_names,
        f1score,
        "F score",
    )

    plot_confusion_matrix_path = f"figures/confusion_matrix_{name_saved}.png"
    plot_confusion_matrix_nor_path = f"figures/confusion_matrix_nor_{name_saved}.png"

    # Plot the confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
    )
    plot_confusion_matrix(
        plot_confusion_matrix_path,
        cm,
        class_names,
        title="Confusion Matrix of Yoga Pose Model",
    )

    plot_confusion_matrix(
        plot_confusion_matrix_nor_path,
        cm,
        class_names,
        normalize=True,
        title="Normalized Confusion Matrix of Yoga Pose Model",
    )

    # Print the classification report
    print(
        "\nClassification Report:\n",
        classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        ),
    )

    with open(f"statistics/classification_report_{name_saved}.txt", "w") as f:
        f.writelines(
            classification_report(
                y_test, y_pred, target_names=class_names, zero_division=0
            )
        )
    f.close()


"""
======================================================== DEF TEST =======================================================
"""
def load_model(model_name):
    newest_model = find_newest_model_with_prefix("models", model_name)
    if model_name not in ["svm"]:
        json_file = open("models/" + newest_model + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("models/" + newest_model + ".h5")
    else:
        with open(f"models/{newest_model}.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    return model

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True, keep_input_size=False
):
    # Draw the detection result on top of the image.
    image_np = utils.visualize(image, [person])

    # Plot the image with detection results.
    # height, width, channel = image.shape
    # aspect_ratio = float(width) / height
    # fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # im = ax.imshow(image_np)

    # if close_figure:
    #     plt.close(fig)

    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np


def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
        [
            [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints
        ],
        dtype=np.float32,
    )
    return pose_landmarks


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


def resizew800(img_arr):
    original_height, original_width, _ = img_arr.shape
    scale = 800 / original_width
    resized_img = cv2.resize(img_arr, None, fx=scale, fy=scale)
    resized_height, resized_width, _ = resized_img.shape
    print(
        f"({original_width}, {original_height}) -> ({resized_width}, {resized_height})"
    )

    return resized_img


def get_skeleton(image):
    person = detect(image)
    pose_landmarks = get_keypoint_landmarks(person)
    # pose_landmarks
    lm_pose = landmarks_to_embedding(
        tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51))
    )

    return person, lm_pose


def predict_pose(model_name, model, lm_pose, class_names):
    if model_name not in ["svm"]:
        predict = model.predict(lm_pose)
    else:
        lm_pose = np.array(lm_pose)
        predict = model.predict_proba(lm_pose)

    class_name_pred = class_names[np.argmax(predict)]
    acc_pred = np.max(predict[0], axis=0)

    print("Class name: ", class_name_pred)
    print("Acurracy: ", acc_pred)

    return class_name_pred, acc_pred, predict


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
