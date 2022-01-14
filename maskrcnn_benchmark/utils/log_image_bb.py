import tensorboardX
import torch
import numpy as np

from maskrcnn_benchmark.data.transforms.transforms import DeNormalize
from torchvision.transforms import ToPILImage
from PIL.ImageDraw import Draw
from PIL import Image, ImageFont
import matplotlib.pyplot as plt
import os

def log_image_and_bb(summary_writer, name, global_step, image, targets, image_name=None, output_dir=None):
    """Logs on the 'summary_writer' with tag 'name' at step 'global_step' the 'image' with the 
    bounding boxes defined in 'targets'

    Arguments:
        summary_writer (TensorboardX summary writer): summary writer on which to write
        name (str): tag for the log
        global_step (int): step for the log
        image (tensor or np array)
        targets (dict or similar): with attributes 'boxes' and 'labels'
    """

    if isinstance(image, torch.Tensor):
        image = image.numpy()

    image = image.astype(np.uint8)

    # bgr to rgb
    image = image[:,:,::-1]
    
    image = _overlay_boxes(image, targets)
    
    image = np.transpose(image, (2,0,1))

    pil_image = Image.fromarray(np.transpose(image,(1,2,0)))

    if output_dir is None:
        if summary_writer is not None:
            logbbdir = summary_writer.logdir + "/bb_output"
        else:
            logbbdir = "./bb_output"
    else:
        logbbdir = output_dir + "/bb_output"


    if not os.path.isdir(logbbdir):
        os.mkdir(logbbdir)

    if image_name is not None:
        pil_image.save(logbbdir + "/" + image_name + '_' + name + '.png')
    else:
        pil_image.save(logbbdir + '/img_' + str(global_step) + '_' + name + '.png')

    if summary_writer is not None:
        summary_writer.add_image(name, image, global_step=global_step)

def _compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    #palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    
    cmap = plt.cm.get_cmap('hsv', 21)

    #color = tuple([int(c * 255) for c in cmap(label)[:3]]) 
    #if not labels.dtype == torch.int64:
    #    palette = palette.float()
    cmcolors = []
    for lbl in labels:
        cmcolors.append(np.array(cmap(lbl)[:3])*255)
    colors = np.array(cmcolors).astype(np.uint8)
    #colors = labels[:, None] * palette.to(labels.device)
    #colors = (colors % 255).cpu().numpy().astype("uint8")
    return colors

def _overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # filter detections by score
    scores = None
    if predictions.has_field("scores"):
        predictions = predictions[predictions.get_field('scores') > 0.5]
        scores = predictions.get_field('scores')

    if predictions.has_field("labels"):
        labels = predictions.get_field("labels")
    else:
        labels = torch.ones(len(predictions.bbox), dtype=torch.float)
    boxes = predictions.bbox

    colors = _compute_colors_for_labels(labels).tolist()

    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', 35)

    pil_image = Image.fromarray(image)
    draw = Draw(pil_image)
    for box, color, score in zip(boxes, colors, scores):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=tuple(color), width=15)

        mtext = "{:.2f}".format(score)
        left_off=7
        top_off=5
        xmin,ymin,xmax,ymax = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        draw.rectangle([xmin, ymin, xmin+100, ymin+40], outline=tuple(color), fill=tuple(color), width=10)
        draw.text((xmin-1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin-1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+1+left_off,ymin+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin+1+top_off), mtext, font=fnt, fill=(0,0,0,128))
        draw.text((xmin+left_off,ymin+top_off), mtext, font=fnt, fill=(255,255,255,128))

    del draw
    return np.array(pil_image)


def log_test_image(cfg, summary_writer, name, global_step, image_list, targets, image_name=None, output_dir=None):
    image_tensor = image_list.tensors[0].cpu()
    image_size = image_list.image_sizes[0]

    targets = targets[0]
    # fix size
    image = image_tensor.clone().detach()[:image_size[1], :image_size[0]].numpy()

    image = np.transpose(image, (1, 2, 0))
    image = DeNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)(image)

    log_image_and_bb(summary_writer, name, global_step, image, targets, image_name)

