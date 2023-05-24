import cv2
import torch
import numpy as np
import random
#
#
from utils.general import non_max_suppression, scale_coords, check_img_size, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load
# from yolov7.models.experimental import attempt_load
# from yolov7.utils.general import set_logging, check_img_size
# from yolov7.utils.torch_utils import select_device


class yolo:
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    # Set the classes to filter by name
    classes_to_filter = ['train', 'person','cell phone']  # You can give a list of classes to filter by name ['train', 'person']

    opt = {
        "weights": "weight/yolov7.pt",  # Path to weights file; default weights are for the YOLOv7 "nano" model
        "yaml": "data/coco.yaml",
        "img-size": 640,  # Default image size
        "conf-thres": 0.25,  # Confidence threshold for inference
        "iou-thres": 0.45,  # NMS IoU threshold for inference
        "device": '0',  # Device to run the model on, e.g., '0' or '0,1,2,3' for GPU, or 'cpu' for CPU
        "classes": classes_to_filter  # List of classes to filter or None
    }
    def run(self):
        with torch.no_grad():
            weights, imgsz = self.opt['weights'], self.opt['img-size']
            from utils.general import set_logging
            set_logging()
            from utils.torch_utils import select_device
            device = select_device(self.opt['device'])
            half = device.type != 'cpu'
            from models.experimental import attempt_load
            model = attempt_load(weights, map_location=device)  # Load FP32 model
            stride = int(model.stride.max())  # Model stride
            from utils.general import check_img_size
            imgsz = check_img_size(imgsz, s=stride)  # Check img_size
            if half:
                model.half()
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

            classes = None
            if self.opt['classes']:
                classes = []
                for class_name in self.opt['classes']:
                    classes.append(names.index(class_name))
            return imgsz , stride , device ,half ,model , names ,colors ,classes


    # Initializing model and setting it for inference


        # Open the webcam
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("Cannot open webcam")

        # Get webcam properties
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize object for writing video output
        output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

        # Counting variables
        class_counts = {class_name: 0 for class_name in self.classes_to_filter}

        while True:
            ret, img0 = video.read()

            if ret:
                img = self.letterbox(img0, imgsz, stride=stride)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=False)[0]

                pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes=classes, agnostic=False)
                t2 = time_synchronized()

                # Reset counts for each frame
                class_counts = {class_name: 0 for class_name in self.classes_to_filter}
                for i, det in enumerate(pred):
                    s = ''
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            class_counts[names[int(c)]] += n  # increment class count
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'

                            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print class counts on the screen
                class_counts_str = ', '.join(
                    [f"{class_name}: {class_counts[class_name]}" for class_name in self.classes_to_filter])
                cv2.putText(img0, class_counts_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Object Detection', img0)
                output.write(img0)

                if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                    break
            else:
                break

        output.release()
        video.release()
        cv2.destroyAllWindows()

