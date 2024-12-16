.PHONY: all train test download_models

all: run

test:
	echo "Test"

setup:
	@echo "Installing required libraries..."
	@python -m pip install -r requirements.txt -q

download_models:
	@if [ ! -d "models" ]; then \
		mkdir models; \
	fi
	@cd models && \
	if [ ! -d "bags" ]; then \
		mkdir bags; \
		cd bags && \
		gdown 1weduKdhWlNULKxt37IMrc9eYrBKUKXI-; \
		cd ..; \
		cd ..; \
		echo "$$(pwd)"; \
	fi

	@cd models && \
	if [ ! -d "segment" ]; then \
		mkdir segment; \
		cd segment && \
		gdown 1xvxbSj73m5ggG7fylMmhbThY50q7Tcv4; \
		cd ..; \
		cd ..; \
		echo "$$(pwd)"; \
	fi

download_videos_4MP: setup
	@if [ ! -d "videos" ]; then \
		mkdir videos; \
	fi
	@cd videos && \
	gdown 16eDkntrqj4cSds_vMw5yVbrSMJvywlu_ -O 1.mp4; \
	gdown 1iSuYd9NUsTjX8o4j7gVyr5V07TPGTEP- -O 2.mp4; \
	gdown 1XZNGJCLEKkxjWoQgkCBXEAL9sNaYGEQ7 -O 3.mp4; \
	gdown 11TZ4ZF8Yi5KnnH9ggtlZMSvmF4VAS9cl -O 4.mp4; \
	gdown 1FpzanPl850x9dILXNAcg9_xYxD6L9iQm -O 5.mp4;

download_demo_videos:
	@if [ ! -d "videos" ]; then \
		mkdir videos; \
	fi
	@cd videos && \
	gdown 1Sd870eOsjVrUiGDaaZ7LirTgM7a_8HEP -O demo.mp4; \


clean: 
	@echo "Removing Models, Frames and videos"
	@rm -rf frames
	@rm -rf videos
	@rm -rf models

test_object_tracker: download_demo_videos download_models
	python modules/tracker.py

test_segment:
	python modules/segmet.py


run: setup
	@if [ ! -d "models" ]; then \
		echo "Models folder does not exist. Running download_models..."; \
		$(MAKE) download_models; \
	else \
		echo "Models folder exists."; \
	fi
	@if [ ! -f "videos/demo.mp4" ]; then \
		echo "demo.mp4 does not exist. Running download_demo_videos..."; \
		$(MAKE) download_demo_videos; \
	else \
		echo "demo.mp4 exists."; \
	fi

	@python main.py


