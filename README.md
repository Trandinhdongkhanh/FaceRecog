# FaceRecog
This is MiAI version

Bây giờ các bạn tạo các thư mục như sau:

Tạo thư mục Dataset trong FaceRecog, trong đó tạo tiếp thư mục FaceData và dưới FaceData là tạo tiếp 2 thư mục raw và processed.

Tạo thư mục Models trong FaceRecog để chờ sẵn tẹo lưu model sau.

Bây giờ các bạn sưu tầm ảnh của 2 người trở lên, mỗi người 10 tấm hình rõ mặt (at least 2 người). Mình ví dụ 2 người tên là NguyenVanA và LeThiB nhé. Các bạn tạo 02 thư mục NguyenVanA và LeThiB trong thư mục raw và copy ảnh của 2 người vào riêng 2 thư mục đó, ảnh của ai vào thư mục của người đó nhé.

Ví dụ cây thư mục của mình để các bạn tham khảo:

![image](https://user-images.githubusercontent.com/95671871/236677556-d5973ce0-56ff-4312-88e7-b341020fc3c0.png)

Các bạn đứng ở thư mục FaceRecog chạy lệnh sau để cài tất cả các thư viện cần thiết:
### `pip install -r requirements.txt`

Với chỗ ảnh mà bạn đã sưu tầm bên trên, có thể là ảnh cả người, bây giờ chúng ta sẽ cắt riêng khuôn mặt ra để train nhé. Các bạn chuyển về thư mục FaceRecog và chạy lệnh :
### `python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25`

Các bạn tải weights pretrain về tại link này : https://drive.google.com/drive/folders/1cz7nJIPT6lLjQKdwvFIAM2er1z7rf4Qx?usp=sharing
Sau khi tải xong về, các bạn copy toàn bộ file tải về vào thư mục Models, chú ý chỉ lấy file, bỏ hết các thư mục như hình bên dưới của mình (không có file facemodel.pkl như bên dưới đâu nhé, mình chụp nhầm chút).

![image](https://user-images.githubusercontent.com/95671871/236677693-681f4c62-1d99-4c92-85f4-a546e346fb45.png)

Bây giờ các bạn chuyển về thư mục FaceRecog nếu đang đứng ở thư mục khác nhé. Sau đó chạy lệnh train:
### `python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000`

Cuối cùng các bạn chạy lệnh:
### `python src/main.py`

Kết quả:

![image](https://user-images.githubusercontent.com/95671871/236677770-c18cd725-f168-4ee9-8eb2-b46e71fe21ca.png)
