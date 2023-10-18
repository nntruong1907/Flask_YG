from tensorflow import keras
from keras.models import Model
from sklearn.svm import SVC
from keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    Flatten,
)


def svm_model(
    decision_function="ovo",
    C=100,
    kernel="rbf",
    gamma="scale",
    degree=1,
    random_state=42,
):
    """
    Tạo mô hình SVM tùy chỉnh.

    Parameters:
        decision_function (string): Hàm quyết định, giá trị mặc định: "ovr" (One-Versus-Rest),
                                    lựa chọn khác: "ovo" (One-Versus-One)
        C (float): Tham số điều chỉnh sự nghiêng của biên quyết định.
                    Giá trị lớn của C sẽ tạo ra một biên quyết định chặt chẽ hơn nhưng có thể dẫn đến việc overfitting.
        kernel (string): Loại kernel sử dụng. giá trị mặc định: "rbf" (Radial Basis Function)
                        lựa chọn khác: "linear", "poly"
        gamma (string hoặc float): Tham số gamma trong hàm kernel, giá trị mặc định: "scale"
                                    lựa chọn khác: "auto", 0.001, 0.01, 0.1, 1
        degree (int): bậc của đa thức đối với kernel "poly", giá trị mặc định: 3
        random_state (int hoặc RandomState instance): đảm bảo tính nhất quán của kết quả.

    Returns:
        model (Model): Mô hình SVM được tạo.

    """
    model = SVC(
        decision_function_shape=decision_function,
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        random_state=random_state,
        probability=True,  # cung cấp xác suất dự đoán
    )

    return model


def fcnn1d_model(
    input_shape,
    num_classes=10,
    num_dense_layers=5,
    dropout_fc=0.2,
    optimizers_func="adam",
    learning_rate=0.0005,
    loss_func="categorical_crossentropy",
):
    """
    Tạo mô hình Fully Connected Neural Network (FCNN)1D tùy chỉnh.

    Parameters:
        input_shape (tuple): Kích thước đầu vào (input shape) của mô hình, ví dụ: (34,).
        num_classes (int): Số lớp đầu ra (output classes).
        num_dense_layers (int): Số lớp Dense trong mô hình.
        dropout_fc (float): Giá trị dropout áp dụng sau mỗi lớp Dense.
        optimizers_func (string): Hàm tối ưu hóa (optimizer) được sử dụng để cập nhật trọng số của mô hình.
        learning_rate (float): Tốc độ học (learning rate) của optimizer. Mặc định là 0.001-->adam.
                                Tốc độ học nằm trong khoảng (0.0001, 0.01)
        loss_function (string): Hàm mất mát được sử dụng cho mô hình.
                        Mặc định là 'categorical_crossentropy' cho bài toán phân loại nhiều lớp

    Returns:
        model (Model): Mô hình FCNN1D được tạo.

    """
    num_units_list = [64, 128, 128, 256, 265, 128, 128, 64]
    if num_dense_layers < len(num_units_list):
        dense_units = num_units_list[:num_dense_layers]
    else:
        # Nếu num_dense_layers lớn hơn số lớp đơn vị được định nghĩa sẵn, thì sẽ sử dụng lặp lại lớp đơn vị cuối cùng.
        num_repeats = num_dense_layers - len(num_units_list)
        dense_units = num_units_list + [num_units_list[-1]] * num_repeats

    input_layer = Input(shape=input_shape)
    x = input_layer

    for units in dense_units:
        x = Dense(units, activation="relu")(x)
        if dropout_fc > 0:
            x = Dropout(dropout_fc)(x)

    output_layer = Dense(num_classes, activation="softmax")(x)

    # Tạo mô hình từ các lớp đã định nghĩa
    model = Model(
        inputs=input_layer, outputs=output_layer, name=f"fcnn1d_{num_dense_layers}_dense"
    )

    if optimizers_func == "adam":
        optimizers_func = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizers_func == "sgd":
        optimizers_func = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizers_func == "rmsprop":
        optimizers_func = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss_func, optimizer=optimizers_func, metrics=["accuracy"])

    return model


def conv1d_model(
    input_shape,
    num_classes=10,
    num_maxpools=1,
    num_conv_layers_per_maxpool=2,
    filter_size=3,
    pool_size=2,
    dropout_cnn=0.2,
    dropout_f=0.5,
    optimizers_func="adam",
    learning_rate=0.0005,
    loss_function="categorical_crossentropy",
):
    """
    Tạo mô hình Convolution 1D tùy chỉnh.

    Parameters:
        input_shape (tuple): Kích thước đầu vào (input shape) của mô hình, ví dụ: (34, 1).
        num_classes (int): Số lớp đầu ra (output classes).
        num_maxpools (int): Số lượng lớp MaxPooling1D trong mô hình.
        num_conv_layers_per_maxpool (int): Số lượng lớp tích chập trước mỗi lớp MaxPooling1D.
        filter_size (int): Kích thước bộ lọc lớp tích chập,
                            xác định chiều dài của cửa sổ trượt trên dữ liệu đầu vào.
        pool_size (int): Kích thước cửa sổ MaxPooling1D,
                            xác định kích thước của khu vực mà lớp MaxPooling1D sẽ giảm điểm ảnh xuống.
        dropout_cnn (float): Xác suất dropout được áp dụng sau mỗi lớp tích chập để ngăn overfitting.
        dropout_f (float): Xác suất dropout được áp dụng sau lớp Flatten để ngăn overfitting.
        optimizers_func (string): Thuật toán tối ưu hóa dùng để cập nhật trọng số mô hình.
                            Mặc định là 'adam', có thể chọn: 'adam', 'sgd' hoặc 'rmsprop'.
        learning_rate (float): Tốc độ học của thuật toán tối ưu hóa. Mặc định là 0.001.
                                Tốc độ học nằm trong khoảng (0.0001, 0.01)
        loss_function (string): Hàm mất mát được sử dụng cho mô hình.
                        Mặc định là 'categorical_crossentropy' cho bài toán phân loại nhiều lớp
    Returns:
        model (Model): Mô hình Convolution 1D được tạo.

    """

    num_filters_list = [32, 64, 128, 256, 128, 64]
    if num_conv_layers_per_maxpool < len(num_filters_list):
        num_filters = num_filters_list[:num_conv_layers_per_maxpool]
    else:
        num_repeats = num_conv_layers_per_maxpool - len(num_filters_list)
        num_filters = num_filters_list + [num_filters_list[-1]] * num_repeats

    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(num_maxpools):
        for filters in num_filters:
            x = Conv1D(filters, filter_size, activation="relu", padding="same")(x)
            if dropout_cnn > 0:
                x = Dropout(dropout_cnn)(x)

        x = MaxPooling1D(pool_size)(x)

    x = Flatten()(x)
    if dropout_f > 0:
        x = Dropout(dropout_f)(x)

    x = Dense(64, activation="relu")(x)
    if dropout_cnn > 0:
        x = Dropout(dropout_cnn)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = keras.Model(
        inputs,
        outputs,
        name=f"conv1d-{num_maxpools}maxpool_{num_conv_layers_per_maxpool}conv_per_maxpool",
    )

    if optimizers_func == "adam":
        optimizers_func = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizers_func == "sgd":
        optimizers_func = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizers_func == "rmsprop":
        optimizers_func = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss_function, optimizer=optimizers_func, metrics=["accuracy"])

    return model
