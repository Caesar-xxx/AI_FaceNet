
from model.SSD import FaceMaskDetection
from model.FACENET import InceptionResnetV1
import torch

import cv2
import numpy as np
import glob
import time

class MaskedFaceRecog:
    def __init__(self):
        # 加载检测模型
        face_mask_model_path = r'weights/face_mask_detection.pb'
        self.ssd_detector = FaceMaskDetection(face_mask_model_path,GPU_ratio=0.1)


        # 加载识别模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 实例化
        self.facenet = InceptionResnetV1(is_train=False,embedding_length=128,num_classes=14575).to(self.device)
        # 从训练文件中加载
        self.facenet.load_state_dict(torch.load(r'./weights/3.Chinese_CASIA_ALL_epoch_150/facenet_best.pt',map_location=self.device))
        self.facenet.eval()


        # 加载其他人的特征，并生成每个人的名称PNG图片（以解决中文显示问题）
        self.name_list,self.known_embedding = self.loadFaceFeats()
        
    
    def loadFaceFeats(self):
        # 记录名字
        name_list = []
        # 输入网络的所有人脸图片
        known_faces_input = []
        # 遍历
        known_face_list = glob.glob('./images/*')
        for face in known_face_list:
            name = face.split('\\')[-1].split('.')[0]
            name_list.append(name)
            # 裁剪人脸
            croped_face = self.getCropedFaceFromFile(face)
            if croped_face is None:
                print('图片：{} 未检测到人脸，跳过'.format(face))
                continue
            # 预处理
            img_input = self.imgPreprocess(croped_face)
            known_faces_input.append(img_input)
        # 转为Nummpy
        faces_input = np.array(known_faces_input)
        # 转tensor并放到GPU
        tensor_input = torch.from_numpy(faces_input).to(self.device)
        # 得到所有的embedding,转numpy
        known_embedding = self.facenet(tensor_input).detach().cpu().numpy()
        
        return name_list,known_embedding

    def getCropedFaceFromFile(self,img_file, conf_thresh=0.5 ):
        
        # 读取图片
        # 解决中文路径问题
        img_ori = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
        
        if img_ori is None:
            return None
        # 转RGB
        img = cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB)
        # 缩放
        img = cv2.resize(img,self.ssd_detector.img_size)
        # 转float32
        img = img.astype(np.float32)
        # 归一
        img /= 255
        # 增加维度
        img_4d = np.expand_dims(img,axis=0)
        # 原始高度和宽度
        ori_h,ori_w = img_ori.shape[:2]
        bboxes, re_confidence, re_classes, re_mask_id = self.ssd_detector.inference(img_4d,ori_h,ori_w)
        for index,bbox in enumerate(bboxes):
            class_id = re_mask_id[index] 
            l,t,r,b = int(bbox[0] * ori_w), int(bbox[1] * ori_h), int(bbox[2] * ori_w), int(bbox[3] * ori_h),

            croped_face = img_ori[t:b,l:r]
            return croped_face

        # 都不满足
        return None

    def imgPreprocess(self,img):
        # 转为float32
        img = img.astype(np.float32)
        # 缩放
        img = cv2.resize(img,(112,112))
        # BGR 2 RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # h,w,c 2 c,h,w
        img = img.transpose((2,0,1))
        # 归一化[0,255] 转 [-1,1]
        img = (img - 127.5) / 127.5
        # 增加维度
        # img = np.expand_dims(img,0)

        return img

    def main(self):
    
        cap = cv2.VideoCapture(0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        png_index = 0
        
        while True:
            start_time = time.time()

            ret,frame = cap.read()
            frame = cv2.flip(frame,1)
            # 转RGB
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 缩放
            img = cv2.resize(img,self.ssd_detector.img_size)
            # 转float32
            img = img.astype(np.float32)
            # 归一
            img /= 255
            # 增加维度
            img_4d = np.expand_dims(img,axis=0)
            bboxes, re_confidence, re_classes, re_mask_id = self.ssd_detector.inference(img_4d,frame_h,frame_w)

            for index,bbox in enumerate(bboxes):
                class_id = re_mask_id[index] 
                
                if class_id == 0:
                    color = (0, 255, 0)  # 戴口罩
                elif class_id == 1:
                    color = (0, 0, 255)  # 没带口罩

                l,t,r,b = max(0,int(bbox[0] * frame_w) ), max(0,int(bbox[1] * frame_h)), int(bbox[2] * frame_w), int(bbox[3] * frame_h),

                # 裁剪人脸
                crop_face = frame[t:b,l:r]
               
                
                # 人脸识别
                
                # 转为float32
                img = crop_face.astype(np.float32)
                # 缩放
                img = cv2.resize(img,(112,112))
                # BGR 2 RGB
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                # h,w,c 2 c,h,w
                img = img.transpose((2,0,1))
                # 归一化[0,255] 转 [-1,1]
                img = (img - 127.5) / 127.5
                # 扩展维度
                img_input = np.expand_dims(img,0)
                # C连续特性
                # img_input = np.ascontiguousarray(img_input)
                # 转tensor并放到GPU
                tensor_input = torch.from_numpy(img_input).to(self.device)
                # 得到embedding
                embedding = self.facenet(tensor_input)
                embedding = embedding.detach().cpu().numpy()
                # print(embedding)
                # 计算距离
                dist_list = np.linalg.norm((embedding-self.known_embedding),axis=1)
                # 最小距离索引
                min_index = np.argmin(dist_list)
                # 识别人名与距离
                pred_name = self.name_list[min_index]
                # 最短距离
                min_dist = dist_list[min_index]
                if min_dist < 1:
                    print(pred_name,min_dist)
                

                cv2.rectangle(frame,(l,t),(r,b),color,2)
            

            fps = 1/ (time.time()- start_time)
            cv2.putText(frame,str(round(fps,2)),(50,50),cv2.FONT_ITALIC,1,(0,255,0),1)

            cv2.imshow('demo',frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
                    

mf = MaskedFaceRecog()
mf.main()