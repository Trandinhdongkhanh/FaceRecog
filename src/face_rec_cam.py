from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream

import facenet
import imutils
import pickle
import align.detect_face
import numpy as np
import cv2
import collections


def main():
    MINSIZE = 20  # kích thước tối thiểu của khuôn mặt
    # threshold: threshold=[th1, th2, th3], th1-3 là ngưỡng của 3 giai đoạn phát hiện khuôn mặt
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709  # hệ số được sử dụng để căn chỉnh hình ảnh
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    # Sử dụng thư viện Tensorflow để tạo một đồ thị tính toán (computation graph)
    # và một phiên làm việc (session) để thực thi các phép tính trên đồ thị đó trên GPU.

    # 'with' để tạo một đồ thị tính toán mới với câu lệnh tf.Graph().as_default().
    # Đồ thị tính toán này sẽ là nơi các phép tính Tensorflow được thực hiện.
    with tf.Graph().as_default():
        # Tạo một phiên làm việc sess mới
        # Khi tính toán trên đồ thị, Tensorflow sẽ sử dụng GPU thay vì CPU để tăng tốc độ tính toán.
        # Thông số per_process_gpu_memory_fraction=0.6 cho phép Tensorflow chỉ sử dụng tối đa 60% bộ nhớ GPU có sẵn trên hệ thống.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Tải model đã được huấn luyện trước của FaceNet.
            # model này được sử dụng để trích xuất đặc trưng (feature extraction) từ ảnh khuôn mặt đầu vào
            # FaceNet là một model deep learning được huấn luyện để sinh ra một không gian đặc trưng cho các khuôn mặt
            # Các vector đặc trưng này có thể được sử dụng để so sánh và nhận dạng khuôn mặt
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Sử dụng TensorFlow để lấy các Tensor inputs và outputs từ model đã được trước đó

            # Tensor input được đặt tên "input:0" trong model
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            # tensor output, chứa các vector nhúng (embeddings) biểu diễn cho các khuôn mặt được trích xuất bởi model facenet.
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # Tensor placeholder được sử dụng trong quá trình training
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # PNet, RNet và ONet là các mạng neural network được sử dụng trong MTCNN để phát hiện khuôn mặt
            # và tạo ra các phát hiện cụ thể (bounding boxes) xung quanh các khuôn mặt đó.

            # PNet (Proposal Network) được sử dụng để tạo ra các phát hiện ở các vị trí khác nhau trên ảnh
            # RNet (Refine Network) được sử dụng để cải thiện độ chính xác của các phát hiện tạo ra bởi PNet
            # ONet (Output Network) được sử dụng để tinh chỉnh kích thước và vị trí của các phát hiện

            # Tạo và trả về các hàm để detect khuôn mặt trong ảnh sử dụng model
            # MTCNN (Multi-task Cascaded Convolutional Networks).
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            # people_detected = set()
            person_detected = collections.Counter()

            # Chỗ này nhóm em sử dụng VideoStream thay vì cv2.VideoCapture bởi vì nó sử dụng một số thủ thuật
            # như luồng video phân tán, khử nhiễu và giảm thiểu độ trễ để đạt hiệu suất tốt hơn so với cv2.VideoCapture.
            cap = VideoStream(src=0).start()

            while (True):
                frame = cap.read()  # Đọc frame từ camera
                frame = imutils.resize(frame, width=600)  # Căn chỉnh lại kích thước frame
                frame = cv2.flip(frame, 1)  # Video từ camera thường sẽ bị ngược nên ta đảo chiều nó lại

                # Hàm detect_face() phát hiện khuôn mặt của ảnh đầu vào vẽ xác định khung vuông quanh khuôn mặt đó
                # Hàm sử dụng các bước sau để phát hiện khuôn mặt:

                # 1. Tạo một pyramid size của ảnh đầu vào theo factor được cung cấp.
                # 2. Áp dụng model pnet để tạo các khuôn mặt dự đoán và các khung vuông (bounding boxes) tương ứng.
                # 3. Áp dụng non-maximum suppression (NMS) để loại bỏ các khung trùng lặp và giữ lại khung có điểm cao nhất.
                # 4. Áp dụng rnet để lọc các khuôn mặt được chọn từ bước trước đó và tạo ra các khung mới cho mỗi khuôn mặt.
                # 5. Áp dụng NMS lần cuối để loại bỏ các khung trùng lặp và trả về các khung và các điểm tương ứng cho mỗi khung đó.

                # Trả về các khung chứa khuôn mặt dưới dạng một mảng numpy.
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                # Mỗi bounding box được biểu diễn bằng tọa độ (x1, y1, x2, y2)
                # trục x được đặt theo chiều ngang của hình ảnh, bắt đầu từ trái qua phải
                # trục y được đặt theo chiều dọc của hình ảnh, từ trên xuống dưới

                # x1 và y1 là toạ độ nằm góc trái trên cùng của bounding box
                # x2 và y2 là tọa độ nằm ở góc phải dưới cùng của bounding box
                # Mở file AnhMinhHoa.png trg Images để hình dung rõ hơn

                faces_found = bounding_boxes.shape[0]
                try:
                    # Nếu phát hiện nhiều hơn một khuôn mặt thì thông báo "Only on face"
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        # trích xuất tọa độ của các đỉnh của bounding box
                        # lấy tất cả các hàng của mảng bounding box và lấy cột từ 0 đến 4 (lấy tọa độ của các đỉnh)
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]  # gán gtri tọa độ x1
                            bb[i][1] = det[i][1]  # gán gtri tọa độ y1
                            bb[i][2] = det[i][2]  # gán gtri tọa độ x2
                            bb[i][3] = det[i][3]  # gán gtri tọa độ y2

                            print(bb[i][3] - bb[i][1])
                            print(frame.shape[0])
                            print((bb[i][3] - bb[i][1]) / frame.shape[0])

                            # (bb[i][3] - bb[i][1]): chiều cao khuôn mặt (k.cách từ y2 -> y1)
                            # frame.shape[0]: chiều cao khung hình đầu vào
                            # Nếu tỉ lệ giữa chiều cao khuôn mặt và khung hình đầu vào > 0.25
                            if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                                # thực hiện cắt khuôn mặt đã được xác định
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                # Scale lại cho phù hợp với model
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                # Chuẩn hóa hình ảnh
                                scaled = facenet.prewhiten(scaled)

                                # chuyển đổi kích thước gốc của ảnh được về kích thước được chỉ định của INPUT_IMAGE_SIZE
                                # scaled: ảnh được scale lại để phù hợp với model.
                                # INPUT_IMAGE_SIZE: kích thước đầu vào của ảnh được sử dụng cho model.
                                # 3: đây là số kênh màu của ảnh (R, G, B).
                                # Tham số -1 dc sử dụng để cho phép hàm reshape tự động tính toán số lượng ảnh dựa
                                # trên kích thước của mảng đầu vào.
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                                # truyền dữ liệu ảnh vào biến images_placeholder của graph để tiến hành nhận dạng khuôn mặt.
                                # Trong feed_dict, có hai keys là images_placeholder và phase_train_placeholder.
                                # images_placeholder là biến placeholder được khai báo trước đó trong graph
                                # để đón nhận dữ liệu đầu vào, còn phase_train_placeholder được sử dụng để
                                # xác định trạng thái của model, ở đây có giá trị là False, tức là không huấn luyện model.
                                # vì trc đó ta đã train model rồi
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                                # tính toán các embedding vector cho ảnh khuôn mặt sử dụng các gtri đầu vào đã đc cung cấp
                                # trong feed_dict. Kết quả trả về là embedding vector của ảnh khuôn mặt,
                                # được lưu trữ trong biến emb_array.
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                                # dự đoán xác suất xảy ra của các lớp (các tên của người trong ảnh khuôn mặt)
                                # dựa trên các vector đặc trưng này.
                                # Kết quả là một mảng numpy chứa xác suất dự đoán của model cho mỗi vector ảnh
                                predictions = model.predict_proba(emb_array)

                                # sử dụng hàm argmax() để lọc ra người có xác suất cao nhất và lưu vào
                                # biến best_class_indices
                                best_class_indices = np.argmax(predictions, axis=1)

                                # đồng thời ta cx lưu lại xác suất dự đoán cao nhất phát hiện được tại thời điểm đó
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]

                                # Sau khi đã xác định được người có xác suất cao nhất ta lấy ra tên người đó
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                # Ve khung mau xanh quanh khuon mat
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20

                                # Neu ty le nhan dang > 0.5 thi hien thi ten
                                if best_class_probabilities > 0.5:
                                    name = class_names[best_class_indices[0]]
                                else:
                                    # Con neu <=0.5 thi hien thi Unknown
                                    name = "Unknown"

                                # Viet text len tren frame
                                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                person_detected[best_name] += 1

                except:
                    pass

                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.stop()
            cv2.destroyAllWindows()
