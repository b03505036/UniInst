from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from adet.config import get_cfg
from detectron2.engine import DefaultPredictor
import argparse
import os

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out', help='absoulte path', required=True)
    parser.add_argument('-cfg', required=True, help='absoulte path')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = process_command()
    #Use the final weights generated after successful training for inference
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.WEIGHTS = os.path.join(args.out, "model_final.pth")
    #Pass the validation dataset
    cfg.DATASETS.TEST = ("coco_2017_val", )

    predictor = DefaultPredictor(cfg)
    data_set = cfg.DATASETS.TEST
    #Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("coco_2017_val", cfg, False, output_dir="/test/")
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")
    #Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)