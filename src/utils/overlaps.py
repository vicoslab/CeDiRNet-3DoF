
import torch
import numpy as np
import cv2
import skimage.draw

def mask_to_rotated_bbox(mask, mask_ids, im_shape, center, estimate_type='minmax', device=None):
    if mask_ids is not None:
        mask_ids = torch.from_numpy(np.array(np.unravel_index(list(mask_ids),im_shape)).T).to(device)
    if estimate_type in ['minmax','3sigma-center','3sigma',None]:
        bbox_polygon, rect = _mask_to_rotated_bbox_using_pca(mask, mask_ids, center, estimate_type)
    elif estimate_type in ['cv','cv2','opencv2','opencv','findContours']:
        bbox_polygon, rect = _mask_to_rotated_bbox_using_cv2(mask)
    else:
        raise Exception('Invalid estimate_type in mask_to_rotated_bbox, allowed only: opencv, minmax, 3sigma and 3sigma-center')

    xx, yy = skimage.draw.polygon(bbox_polygon[:, 0], bbox_polygon[:, 1], shape=im_shape)

    bbox_ids = np.ravel_multi_index((xx, yy), dims=im_shape)
    if mask is not None:
        bbox_mask = torch.zeros_like(mask)
        bbox_mask[xx, yy] = 1
    else:
        bbox_mask = None

    return bbox_polygon, bbox_mask, bbox_ids

def _mask_to_rotated_bbox_using_cv2(mask):
    _, contours, _ = cv2.findContours(mask.cpu().numpy(), cv2.RETR_EXTERNAL, 1)
    rect = cv2.minAreaRect(np.concatenate(contours,axis=0).squeeze())
    (x, y), (w, h), a = rect

    bbox_polygon = cv2.boxPoints(rect)
    bbox_polygon = np.concatenate((bbox_polygon,bbox_polygon[:1,:]),axis=0)

    return bbox_polygon[:,::-1], rect

def _mask_to_rotated_bbox_using_pca(mask_, mask_idx, center, estimate_type='minimax'):
    if mask_idx is None:
        mask_idx = torch.nonzero(mask_)
    mask_idx = mask_idx.float()
    mask_idx_center = mask_idx.mean(dim=0)
    A = torch.svd(mask_idx - mask_idx_center)

    if estimate_type == 'minmax':
        US_min = (A.U * A.S).min(dim=0)[0].cpu().numpy()
        US_max = (A.U * A.S).max(dim=0)[0].cpu().numpy()
    elif estimate_type == '3sigma-center':

        pred_center = torch.from_numpy(np.array(center)).float().to(mask_idx.device)
        pred_center_uv = np.matmul((pred_center - mask_idx_center).cpu().reshape(1, -1),
                                   A.V.t().inverse().cpu().numpy())

        stdiv_range = 3 * (A.U * A.S - pred_center_uv.to(A.V.device)).abs().std(dim=0)
        US_max = stdiv_range + pred_center_uv[0].to(A.V.device)
        US_min = -stdiv_range + pred_center_uv[0].to(A.V.device)
    elif estimate_type == '3sigma' or estimate_type == None:
        US_max = 3 * (A.U * A.S).abs().std(dim=0).cpu().numpy()
        US_min = -US_max
    else:
        raise Exception('Invalid estimate_type in _mask_to_rotated_bbox_using_pca, allowed only: minmax, 3sigma and 3sigma-center')

    bbox = np.array([[US_min[0], US_min[1]],
                     [US_min[0], US_max[1]],
                     [US_max[0], US_max[1]],
                     [US_max[0], US_min[1]],
                     [US_min[0], US_min[1]]])

    bbox_polygon = np.matmul(bbox, A.V.cpu().numpy().T) + mask_idx_center.cpu().numpy()

    rect = cv2.minAreaRect(bbox_polygon)

    return bbox_polygon, rect

def overlap_pixels(instance_mask, gt_mask):
    return (torch.sum(instance_mask & gt_mask).type(torch.float32) / torch.sum(instance_mask | gt_mask).type(torch.float32)).item()

def overlap_pixels_px_missing(instance_mask, gt_mask):
    return (torch.sum(instance_mask | gt_mask).type(torch.float32) - torch.sum(instance_mask & gt_mask).type(torch.float32)).item()

def overlap_pixels_ids(instance_ids, gt_mask_ids):
    instance_ids,gt_mask_ids = set(instance_ids), set(gt_mask_ids)
    inter = len(instance_ids.intersection(gt_mask_ids))
    return float(inter) / float(len(instance_ids) + len(gt_mask_ids) - inter)

def overlap_rot_bbox(instances_points, gts_points):
    def _to_rot_bbox_array(pts):
        rb = [cv2.minAreaRect(p) for p in pts]
        return np.array([[cx,cy,h,w,-a] for (cx,cy),(w,h),a in rb])

    return rbbx_overlaps(_to_rot_bbox_array(instances_points),
                         _to_rot_bbox_array(gts_points))

def rbbx_overlaps(boxes, query_boxes):
    '''
    Parameters
    ----------------
    boxes: (N, 5) --- x_ctr, y_ctr, height, width, angle
    query: (K, 5) --- x_ctr, y_ctr, height, width, angle
    ----------------
    Returns
    ----------------
    Overlaps (N, K) IoU
    '''

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        query_area = query_boxes[k, 2] * query_boxes[k, 3]
        for n in range(N):
            box_area = boxes[n, 2] * boxes[n, 3]
            # IoU of rotated rectangle
            # loading data anti to clock-wise
            rn = ((boxes[n, 0], boxes[n, 1]), (boxes[n, 3], boxes[n, 2]), -boxes[n, 4])
            rk = (
            (query_boxes[k, 0], query_boxes[k, 1]), (query_boxes[k, 3], query_boxes[k, 2]), -query_boxes[k, 4])
            int_pts = cv2.rotatedRectangleIntersection(rk, rn)[1]
            # print type(int_pts)
            if None is not int_pts:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                overlaps[n, k] = int_area * 1.0 / (query_area + box_area - int_area)
    return overlaps
