  import depthai as dai
  import cv2

  class OakCamera:
      def __init__(self):
          self.pipeline = dai.Pipeline()
          # ... (setup code as in previous message)
      def get_frames(self):
          with dai.Device(self.pipeline) as device:
              q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
              q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
              while True:
                  in_rgb = q_rgb.get()
                  in_depth = q_depth.get()
                  frame = in_rgb.getCvFrame()
                  depth_frame = in_depth.getFrame()
                  yield frame, depth_frame
