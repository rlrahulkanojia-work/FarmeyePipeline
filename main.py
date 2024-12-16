"""Main Module for Farm Module"""

import os
import cv2

from modules.tracker import ObjectTracker
from modules.segmet import ZoneDetector
from modules.sort import Sort

# TODO: Reshape masks as per image size

class FarmPipeline:

    def __init__(self):
        self.objectTracker = ObjectTracker(
            model_path="models/bags/farm_pipeline_0.1.pt",
            video_path="videos/demo.mp4"
        )

        self.zoneDetector = ZoneDetector(
            model_path="models/segment/yolov5s_50_03.pt"
        )


        self.zone_trigger = 1000
        self.zone = None


        #.... Initialize SORT .... 
        self.sort_max_age = 15 
        self.sort_min_hits = 10
        self.sort_iou_thresh = 0.05
        self.sort_tracker = Sort(max_age=self.sort_max_age,
                            min_hits=self.sort_min_hits,
                            iou_threshold=self.sort_iou_thresh) 
        self.track_color_id = 0


        ## Video Params
        self.video_path = "videos/demo.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = 0

        self.COUNTER = 0

        self.save_video = True
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.save_video == True:
            self.out = cv2.VideoWriter(
                "videos/output.mp4",
                self.fourcc,
                20.0,
                (1280, 732))  # Initialize VideoWriter


    
    def update_zone(self, image):
        self.zone = self.zoneDetector.inference(image)
        self.zone = cv2.resize(self.zone, (1280, 732), interpolation=cv2.INTER_LINEAR)
        self.zone_colored = cv2.cvtColor(self.zone, cv2.COLOR_GRAY2BGR) * 255

    def run(self):

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            
            # Break the loop if there are no more frames
            if not ret:
                print("End of video or error occurred.")
                break
            
            if frame_count % self.zone_trigger == 0:
                self.update_zone(frame)

            # Increment the frame count
            frame_count += 1

            dets_to_sort, status = self.objectTracker.image_inference(frame, self.zone)

            # Run SORT
            tracked_dets, deleted_trackers_info = self.sort_tracker.update(dets_to_sort)
            tracks = self.sort_tracker.getTrackers()
            frame = self.objectTracker.visualize(frame, tracked_dets, tracks)

            if len(deleted_trackers_info)>0:
                print("deleted_trackers_info" , deleted_trackers_info)
                for tracker in deleted_trackers_info:
                    if self.zone[tracker[1], tracker[0]] == 1:
                        self.COUNTER += 1
                        print("Counter : ", self.COUNTER)

            frame = cv2.putText(frame,  f"Counter: {self.COUNTER}", (600, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Overlay the zone on the frame with 0.5 weightage
            frame = cv2.addWeighted(frame, 0.8, self.zone_colored, 0.5, 0)
            
            if self.objectTracker.view_img:
                cv2.imshow(str("sa"), frame)
                cv2.waitKey(1) 

            if self.save_video:
                self.out.write(frame)


        if self.save_video:
            self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    farm_pipeline = FarmPipeline()
    farm_pipeline.run()