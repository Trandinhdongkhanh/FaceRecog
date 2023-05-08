import cv2
import time


def main():
    cap = cv2.VideoCapture('video/BanKhanh.mp4')
    # Resolution 640*480
    time.sleep(1)
    if cap is None or not cap.isOpened():
        print('Khong the mo file video')
        return
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE);
    n = 1
    dem = 200
    i = 1
    while i <= 100:
        [success, img] = cap.read()
        ch = cv2.waitKey(30)
        if success:
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            imgROI = img[40:(40 + 480), :]  # Tao ra anh 480x480

            imgROI = cv2.resize(imgROI, (250, 250))
            cv2.imshow('Image', imgROI)
        else:
            break
        if n % 4 == 0:
            filename = 'Dataset/FaceData/raw/BanKhanh/BanKhanh%04d.bmp' % (dem)
            cv2.imwrite(filename, imgROI)
            dem = dem + 1
            i = i + 1
        n = n + 1
    return


if __name__ == "__main__":
    main()
