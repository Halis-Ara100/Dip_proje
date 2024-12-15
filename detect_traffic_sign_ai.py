
import argparse, os, sys, pathlib
from pathlib import Path
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import LOGGER,Profile,cv2,non_max_suppression,print_args,scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5/best_traffic_sign_15_11.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml", 
    imgsz=(416, 416),  
    conf_thres=0.25,  
    iou_thres=0.45, 
    max_det=1,  
    device="0", 
    classes=None, 
    line_thickness=3,  
    hide_labels=False, 
    hide_conf=False, 
    vid_stride=1,
):

    source = "0"
    webcam = source.isnumeric() or source.endswith(".streams")
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    bs = 8
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=False, visualize=False).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)
            s += "%gx%g " % im.shape[2:]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1) 

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt) 
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "C:/Users/hsrkv/OneDrive/Masaüstü/vscode/target_local/yolov5/best_traffic_sign_15_11.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[416,416], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1, help="maximum detections per image")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

opt = parse_opt()
run(**vars(opt))