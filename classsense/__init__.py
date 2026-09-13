# classsense - classroom engagement monitoring.
#
# Import order note: config has no internal dependencies, geometry depends only
# on config, and everything else builds upward from those two. Keeping that
# direction one-way is what lets the training scripts import geometry without
# dragging in OpenCV windows, MediaPipe pools or YOLO.

__version__ = "0.2.0"
