# cd /Users/x20016xx/userbility/face_recognition
# source ./bin/activate
# python sampleWithGUI_04.py


# ステップ1. インポート
import PySimpleGUI as sg  # PySimpleGUIをsgという名前でインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import numpy as np  # numpyのインポート
import cv2  # OpenCV（python版）のインポート
import dlib  #Dlibのインクルード

# ---------  関数群  ----------

# モザイク処理関数
def mosaic(src, ratio=0.1):
	small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
	return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
	# src.shape[:2][::-1]は要素を先頭から2つ取ってきて逆順に並び替える操作
	# nparrayは行・列の並びなので，例えば画像サイズが（640,480）の時
	# shapeは shape[0]=480, shape[1]=640（480, 640）となる．
	# resizeに渡す第２引数は（横640，縦480）なので，[::-1]で逆順に並べ替える
	# （参考）https://qiita.com/tanuk1647/items/276d2be36f5abb8ea52e

def mozaic_area(src, box, ratio=0.1):
	dst = src.copy()
	# 画像外へのアクセスチェック
	if box[0] < 0:
		box[0] = 0;
	if box[1] < 0:
		box[1] = 0;
	if box[2] >= src.shape[1]:
		box[2] = src.shape[1];
	if box[3] >= src.shape[0]:
		box[3] = src.shape[0];
	dst[box[1]:box[3], box[0]:box[2]] = mosaic(dst[box[1]:box[3], box[0]:box[2]], ratio)
	
	return dst

# アイコンを読み込む関数
def load_icon(path, distance):
	icon = cv2.imread(path, -1)
	icon_height, _  = icon.shape[:2]
	icon = img_resize(icon, float(distance * 1.5/icon_height))
	icon_h, icon_w  = icon.shape[:2]
	
	return icon, icon_w, icon_h

# 画像をリサイズする関数
def img_resize(img, scale):
	h, w  = img.shape[:2]
	img = cv2.resize(img, (int(w*scale*0.45), int(h*scale*0.45)))
	return img

# 距離と顔の中心座標を計算
def calc_distance_and_pos(img, parts, isLandmarkON):
	# 確認(33が顔の中心位置)
	cnt = 0
	pos = None
	p1 = None
	distance = 0.0
	
	for i in parts:
		if (cnt == 0):
			# 顔の幅を測る時の始点
			p1 = i
		if (cnt == 14):
			# 顔の幅を計算
			distance = ((p1.x-i.x)**2 + (p1.y-i.y)**2)**0.5
		if (cnt == 29):
			pos = i # 顔の中心位置
		if (isLandmarkON == 'あり'):
			# 画像に点とテキストをプロット
			cv2.putText(img, str(cnt), (i.x, i.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
			cv2.circle(img, (i.x, i.y), 1, (255, 0, 0), -1)
		cnt = cnt + 1
	
	return distance, pos

# 画像を保存する関数
def save_image(img):
	date = datetime.now().strftime("%Y%m%d_%H%M%S")
	path = "./" + date + ".png"
	cv2.imwrite(path, img) # ファイル保存

# 画像を合成する関数
def merge_images(bg, fg_alpha, s_x, s_y):
	alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
	alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
	alpha = alpha / 255.0    # 0.0〜1.0の値に変換
	
	fg = fg_alpha[:,:,:3]
	
	f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
	b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得
	
	# 画像の大きさと開始座標を表示
	print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))
	
	bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
	bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成
	
	return bg
	
# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
	h, w = img.shape[:2]
	width = round(w * (height / h))
	dst = cv2.resize(img, dsize=(width, height))
	
	return dst

# ---- 顔認識エンジンセット ----
detector = dlib.get_frontal_face_detector()  #Dlibに用意されている検出器をセット
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 学習済みファイル読み込み

# ---- 大域変数 ----
display_size = (400, 300)  # ディスプレイサイズ
isOpened = 0  # カメラがオープンになっているかどうかのフラグ
isRun = 0  # 認識On/Offフラグ（Onの時に認識実行）
IMAGE_PATH = "./megane/AUCF-22A-181_86.png"  # 画像パス
IMAGE_PATH2 = "./megane/AMTF-17A-047_97.png"  # 画像パス
IMAGE_PATH3 = "./megane/AMCF-22A-229_127.png"  # 画像パス
IMAGE_PATH4 = "./megane/AUMF-22A-171_94.png"  # 画像パス


# ---- ここからGUIの設定込みの顔画像認識 ----
# ステップ2. デザインテーマの設定
sg.theme('DarkGreen5')

# ステップ3. ウィンドウの部品とレイアウト
layout = [
		  [sg.Text('1:カメラを起動')],[sg.Text('2:任意のメガネを選択')],
		  [sg.Button('カメラを起動', key='camera')],
		  [sg.Image(filename='', size=display_size, key='-input_image-')],
		  [sg.Text('ランドマーク', size=(15, 1)), sg.Combo(('あり', 'なし'), default_value='なし', size=(5, 1), key='landmark'),
		  		  sg.Text('メガネ表示', size=(10, 1)), sg.Combo(('AUCF-22A-181_86', 'AMTF-17A-047_97', 'AMCF-22A-229_127', 'AUMF-22A-171_94'), default_value='AUCF-22A-181_86', size=(15, 1), key='megane')],
		  [sg.Button('開始/外す', key='run'), sg.Button('終了', key='exit')],
		  ]

# ステップ4. ウィンドウの生成
window = sg.Window('メガネ試着ツール', layout, location=(400, 20))

# ステップ5. イベントループ
while True:
	event, values = window.read(timeout=10)
	
	if event in (None, 'exit'): #ウィンドウのXボタンまたは”終了”ボタンを押したときの処理
		break
	
	if event == 'read': #「読み込み」ボタンが押されたときの処理
		# 画像の読み込み処理
		print("Read image = " + values['inputFilePath'])
		orig_img = cv2.imread(values['inputFilePath'])  # OpenCVの画像読み込み関数を利用
		# 画像が大きいのでサイズを1/2にする．shapeに画像のサイズ(row, column)が入っている
		#		height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
		#		width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）
		#		orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))  #OpenCVのresize関数
		# 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
		disp_img = scale_to_height(orig_img, display_size[1])
		# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
		imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
		# ウィンドウへ表示
		window['-input_image-'].update(data=imgbytes)
	
	if event == 'camera':  #「カメラ」ボタンが押された時の処理
		print("Camera Open")
		cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
		isOpened, orig_img = cap.read()
		if isOpened:  # 正常にフレームを読み込めたら
			fps = cap.get(cv2.CAP_PROP_FPS)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			orig_img = cv2.resize(orig_img , (int(width/2), int(height/2)))
			height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
			width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）

			print("Frame size = " + str(orig_img.shape))
			# 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
			disp_img = scale_to_height(orig_img, display_size[1])
			# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
			imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
			# ウィンドウへ表示
			window['-input_image-'].update(data=imgbytes)
		else:
			print("Cannot capture a frame image")
			
	if event == 'run':  #「認識開始」ボタンが押された時の処理
		isRun = (isRun + 1) % 2
		if isRun:
			print("Start recognition")
		else:
			print("Finish recognition")
			
	if isOpened == 1:
		# ---- フレーム読み込み ----
		ret, orig_img = cap.read()
		if ret:  # 正常にフレームを読み込めたら
			orig_img = cv2.resize(orig_img , (int(orig_img.shape[1]/2), int(orig_img.shape[0]/2)))
			height = orig_img.shape[0]  #shape[0]は行数（画像の縦幅）
			width = orig_img.shape[1]  #shape[1]は列数（画像の横幅）
			if isRun:  # 認識実行中なら
				# ---- 顔認識実行 ----
				img = orig_img  #処理結果表示用の変数 img を用意して，orig_imgをコピー
				faces = detector(orig_img[:, :, ::-1])  #画像中の全てを探索して「顔らしい箇所」を検出
				if len(faces) > 0:  # 顔を見つけたら以下を処理する
					for face in faces:  #全ての顔 faces から一つの face を取り出して

						parts = predictor(orig_img, face).parts()  #顔パーツ推定
						distance, pos = calc_distance_and_pos( orig_img, parts, values['landmark'])
						
						if (values['megane'] == 'AUCF-22A-181_86'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img
						
						if (values['megane'] == 'AMTF-17A-047_97'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH2, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img

						if (values['megane'] == 'AMCF-22A-229_127'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH3, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*3)/4)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img

						if (values['megane'] == 'AUMF-22A-171_94'):
							icon, icon_w, icon_h = load_icon(IMAGE_PATH4, distance)
							if (pos != None) and (distance > 0.0):
								# icon画像の中心をランドマーク33番（鼻）に対応づける
								x = pos.x - int(icon_w/2)
								y = pos.y - int((icon_h*2)/3)
								if (0 <= y) and (y <= (height-int(icon_h))) and (0 <= x) and (x <= (width-int(icon_w))):
									# 画面の範囲内だったら画像を合成
									img = merge_images(img, icon, x, y)
									outimg = img

				# 表示用に画像を固定サイズに変更
				disp_img = scale_to_height(outimg, display_size[1])
				# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
				imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
				# ウィンドウへ表示
				window['-input_image-'].update(data=imgbytes)
			else:  # 認識フラグOffなら入力フレームをそのまま表示
				# 表示用に画像を固定サイズに変更
				disp_img = scale_to_height(orig_img, display_size[1])
				# 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
				imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
				# ウィンドウへ表示
				window['-input_image-'].update(data=imgbytes)
			
	if event == 'save': #「画像保存」ボタンが押されたときの処理
		print("Write image -> " + values['outputFile'])
		cv2.imwrite(values['outputFile'], outimg)  # OpenCVの画像書き出し関数を利用
			
window.close()

