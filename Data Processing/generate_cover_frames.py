import json, os, shutil
import cv2, math
import multiprocessing

# extract frames from each video and save to the output_folder according to the frame number and frame interval (execution function)
def get_frames_from_video(video_path, video_list, frame_no, frame_interval): 

	output_folder = 'xxx_frames/' # to input

	if not os.path.exists(output_folder):

		os.mkdir(output_folder)

	current_files = os.listdir(output_folder)

	for video in video_list:

		if str(video)+'-'+str(frame_no)+'.jpg' in current_files:

			continue

		video_capture = cv2.VideoCapture(video_path+video+'.mp4')
		fps = video_capture.get(cv2.CAP_PROP_FPS)
		frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

		anchor_point = int(frames/2)

		if frame_no%2 == 1:
			frame_index_list = [anchor_point,]
			half_remain_frames = int((frame_no - 1)/2)
			cut_count = 0
			while cut_count < half_remain_frames:
				cut_count += 1
				frame_index_list.append(anchor_point-cut_count*frame_interval)
				frame_index_list.append(anchor_point+cut_count*frame_interval)
		else:
			frame_index_list = [int(anchor_point-frame_interval/2), math.ceil(anchor_point+frame_interval/2)]
			half_remain_frames = int((frame_no - 2)/2)
			cut_count = 0
			while cut_count < half_remain_frames:
				cut_count += 1
				frame_index_list.append(int(anchor_point-frame_interval/2)-cut_count*frame_interval)
				frame_index_list.append(math.ceil(anchor_point+frame_interval/2)+cut_count*frame_interval)
		frame_index_list = sorted(frame_index_list)

		### complete all frames If there are not enough frames due to short video
		if len(frame_index_list) < frame_no:
			front_frame, rear_frame = frame_index_list[0], frame_index_list[-1]
			while True:
				frame_index_list.append(front_frame)
				if len(frame_index_list) == frame_no:
					break

				frame_index_list.append(rear_frame)
				if len(frame_index_list) == frame_no:
					break
			frame_index_list = sorted(frame_index_list)

		i = 1
		frame_count = 1
		while True:
			success, image_tensor = video_capture.read()
			i += 1
			if i in frame_index_list:
				cv2.imwrite(output_folder+str(video)+'-'+str(frame_count)+'.jpg', image_tensor)
				frame_count += 1
			if not success or i == frame_index_list[-1]:
				break


		### extract all frames If the video is incomplete
		if (len(frame_index_list)+1) != frame_count:
			frame_index_list = [10+frame_interval*k for k in range(frame_no)]

			if len(frame_index_list) < frame_no:
				front_frame, rear_frame = frame_index_list[0], frame_index_list[-1]
				while True:
					frame_index_list.append(front_frame)
					if len(frame_index_list) == frame_no:
						break

					frame_index_list.append(rear_frame)
					if len(frame_index_list) == frame_no:
						break
				frame_index_list = sorted(frame_index_list)

			i = 1
			frame_count = 1
			video_capture = cv2.VideoCapture(video_path+video+'.mp4')
			success, image_tensor = video_capture.read()
			while True:
				success, image_tensor = video_capture.read()
				i += 1
				if i in frame_index_list:
					cv2.imwrite(output_folder+str(video)+'-'+str(frame_count)+'.jpg', image_tensor)
					frame_count += 1
				if not success or i == frame_index_list[-1]:
					break

# extract frames from each video and save to the output_folder according to the frame number and frame interval (main function)
def generate_video_frames(process_no, video_path, frame_no, frame_interval):

	video_list = os.listdir(video_path)

	multi_list = [video_list[i:i + int(len(video_list)/process_no)] for i in range(0, len(video_list), int(len(video_list)/process_no))]
	if len(multi_list) != process_no:
		multi_list[-2] += multi_list[-1]
		del multi_list[-1]

	getKeyProcessList = []
	for i in range(process_no):
		process = multiprocessing.Process(target = get_frames_from_video, args = (video_path, multi_list[i], frame_no, frame_interval))
		process.start()
		getKeyProcessList.append(process)
	for p in getKeyProcessList:
		p.join()
		print("OK!")
	print('frames ready')


if __name__ == '__main__':


	# # frame_no: 1 5 10 (fix frame interval as 5); frame_interval: 1 5 25 (fix frame number as 5); anchor_point: middle point of video
	video_path = 'XXX_VIDEOS/' # source path
	process_no = 10 # cpu core number
	# for frame_interval in [1, 5, 25]:
	for frame_interval in [1]:
		frame_no = 5
		generate_video_frames(process_no, video_path, frame_no, frame_interval)