"""Object detector and Tracker """
import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from modules.sort import *
except:
    from sort import *


class ObjectTracker:

    def __init__(
            self,
            model_path="models/bags/farm_pipeline_0.1.pt",
            video_path="videos/demo.mp4"
        ):
        # Model
        self.model_path = model_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        self.names = self.model.names
        self.model.conf=0.45
        self.model.to("mps")

        #.... Initialize SORT .... 
        self.sort_max_age = 5 
        self.sort_min_hits = 2
        self.sort_iou_thresh = 0.2
        self.sort_tracker = Sort(max_age=self.sort_max_age,
                            min_hits=self.sort_min_hits,
                            iou_threshold=self.sort_iou_thresh) 
        self.track_color_id = 0

        ## Video Params
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = 0
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        if not self.cap.isOpened:
            print("Invalid Video.. Existing")
            exit(1)

        self.view_img = True
        self.color_box =False 

    def compute_color_for_labels(self, label):
        color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    @staticmethod
    def bbox_rel(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """

        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def draw_boxes(self, img, bbox, identities=None, categories=None, 
                    names=None, color_box=None,offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
            label = str(id)

            if color_box:
                color = self.compute_color_for_labels(id)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
                cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                [255, 255, 255], 1)
                cv2.circle(img, data, 3, color,-1)
            else:
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
                cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                [255, 255, 255], 1)
                cv2.circle(img, data, 3, (255,191,0),-1)
        return img

    def visualize(self, frame, tracked_dets, tracks):

        #loop over tracks
        for track in tracks:
            if self.color_box:
                color = self.compute_color_for_labels(track_color_id)
                [cv2.line(frame, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                        color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
                        if i < len(track.centroidarr)-1 ] 
                track_color_id = track_color_id+1
            else:
                [cv2.line(frame, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                        (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                        if i < len(track.centroidarr)-1 ] 

        # draw boxes for visualization
        if len(tracked_dets)>0:
            bbox_xyxy = tracked_dets[:,:4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            self.draw_boxes(frame,
                            bbox_xyxy,
                            identities,
                            categories,
                            self.names,
                            self.color_box)

        return frame

    def run(self):
        
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            
            # Break the loop if there are no more frames
            if not ret:
                print("End of video or error occurred.")
                break
            
            # Increment the frame count
            frame_count += 1

            preds = self.model(frame)
            dets_to_sort = np.empty((0,6))
            for pred in preds.xyxy:
                for x1,y1,x2,y2,conf,detclass in pred.cpu().detach().numpy():
                    dets_to_sort = np.vstack(
                        (
                            dets_to_sort, 
                            np.array([x1, y1, x2, y2, conf, detclass])
                        )
                    )

            # Run SORT
            tracked_dets, _ = self.sort_tracker.update(dets_to_sort)
            tracks = self.sort_tracker.getTrackers()
            frame = self.visualize(frame, tracked_dets, tracks)
        
            if self.view_img:
                cv2.imshow(str("sa"), frame)
                cv2.waitKey(1) 

        # Release the video capture object and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def image_inference(self, image, mask):
        preds = self.model(image)
        status = []
        dets_to_sort = np.empty((0,6))
        for pred in preds.xyxy:
            for x1,y1,x2,y2,conf,detclass in pred.cpu().detach().numpy():
                dets_to_sort = np.vstack(
                    (
                        dets_to_sort, 
                        np.array([x1, y1, x2, y2, conf, detclass])
                    )
                )
                # status.append(mask[(x1 + x2) / 2 , (y1 + y2) / 2 ])
        
        return dets_to_sort, status

            
if __name__ == "__main__":
    object_tracker = ObjectTracker()
    object_tracker.run()
        