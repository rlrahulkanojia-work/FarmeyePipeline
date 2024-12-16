# FarmPipeline

Farm Projects

example command: 
- `python scripts/extract_images.py videos/4MP/1.mp4 frames/4MP/frames/1MP4 --start_time 11:00 --end_time "end" --fps 3`


Initial Setup

1. Create Conda environment : `conda create -n farm python=3.9`
2. `make run` :  To run pipeline on demo video.
3. `make test_object_tracker` : To Test object tracker code.
4. `make test_segment` : To run zone (Truck entrance) Segmenter.
5. `make download_videos_4MP` : Download full length videos.
