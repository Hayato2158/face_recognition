#
# dlibとOpenCVが使えるかどうかチェック
#
import cv2
import dlib

def main():
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 学習済みファイル読み込み

	img = cv2.imread('image.jpg')

	cv2.imshow("check", img) # 画像を表示
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
