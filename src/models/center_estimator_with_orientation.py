import torch

from models.center_estimator import CenterEstimator
from models.center_estimator_fast import CenterEstimatorFast

class CenterOrientationEstimator(CenterEstimator):
    def __init__(self, args=dict(), is_learnable=True):
        super(CenterOrientationEstimator, self).__init__(args, is_learnable=is_learnable)

        self.enable_6dof = args.get('enable_6dof')
        self.use_orientation_confidence_score = args.get('use_orientation_confidence_score')

    def init_output(self, num_vector_fields=1):
        super(CenterOrientationEstimator, self).init_output(num_vector_fields)

        REQUIRED_VECTOR_FIELDS = 5
        if self.enable_6dof:
            REQUIRED_VECTOR_FIELDS += 4 
        if self.use_orientation_confidence_score:
            REQUIRED_VECTOR_FIELDS += 1
        
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

    def forward(self, input, ignore_gt=False, **gt):
        ret = super(CenterOrientationEstimator, self).forward(input, ignore_gt, **gt)

        # use input from parent forward (in case this is modified)
        input = ret['output']
        center_pred = ret['center_pred']

        predictions = input[:, 0:self.num_vector_fields]

        batch_size = center_pred.shape[0]
        num_pred = center_pred.shape[1]

        # WARNING: this assumes CenterEstimator is used as parent WITHOUT fourier (!!)
        if self.enable_6dof:
            sin_orientation = predictions[:, 3:6]
            cos_orientation = predictions[:, 6:9]
        else:
            sin_orientation = predictions[:, 3:4]
            cos_orientation = predictions[:, 4:5]

        prediction_angles = torch.zeros((batch_size,num_pred,sin_orientation.shape[1]))

        if self.use_orientation_confidence_score:
            prediction_confidence_score = predictions[:, 9:10] if self.enable_6dof else predictions[:, 5:6]
            orientation_confidence_score = torch.zeros((batch_size,num_pred,1)).to(center_pred.device)

        for b in range(batch_size):
            # for every predicted center point find its center
            for i, pred in enumerate(center_pred[b]):
                if pred[0] != 0:
                    x,y = pred[1:3]
                    s = sin_orientation[b,:,int(y), int(x)]
                    c = cos_orientation[b,:, int(y), int(x)]
                    pred_angle = torch.atan2(c, s)

                    pred_angle = torch.rad2deg(pred_angle)
                    pred_angle += 360 * (pred_angle < 0).int()

                    prediction_angles[b,i,:] = pred_angle

                    if self.use_orientation_confidence_score:
                        orientation_confidence_score[b,i] = prediction_confidence_score[b,0,int(y), int(x)]

        # add orientations to list of returned values
        ret['pred_angle'] = prediction_angles

        # add orientation confidence score to predictions if needed
        if self.use_orientation_confidence_score:
            ret['center_pred'] = torch.cat((center_pred,orientation_confidence_score),axis=2)

        return ret
    

import time

class CenterOrientationEstimatorFast(CenterEstimatorFast):
    def __init__(self, args=dict(), is_learnable=True):
        super(CenterOrientationEstimatorFast, self).__init__(args, is_learnable=is_learnable)

        self.enable_6dof = args.get('enable_6dof')        
        self.use_orientation_confidence_score = args.get('use_orientation_confidence_score')

    def init_output(self, num_vector_fields=1):
        super(CenterOrientationEstimatorFast, self).init_output(num_vector_fields)

        REQUIRED_VECTOR_FIELDS = 5
        if self.enable_6dof:
            REQUIRED_VECTOR_FIELDS += 4 
        if self.use_orientation_confidence_score:
            REQUIRED_VECTOR_FIELDS += 1
        
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS


    def forward(self, input, ignore_gt=False, **gt):
        center_pred, times = super(CenterOrientationEstimatorFast, self).forward(input)

        start_orient = time.time()
        predictions = input[:, 0:self.num_vector_fields]

        num_pred = center_pred.shape[0]

        # WARNING: this assumes CenterEstimator is used as parent WITHOUT fourier (!!)
        if self.enable_6dof:
            sin_orientation = predictions[:, 3:6]
            cos_orientation = predictions[:, 6:9]
        else:
            sin_orientation = predictions[:, 3:4]
            cos_orientation = predictions[:, 4:5]

        if self.use_orientation_confidence_score:
            prediction_confidence_score = predictions[:, 9:10] if self.enable_6dof else predictions[:, 5:6]
            orientation_confidence_score = torch.zeros((num_pred,1)).to(center_pred.device)

        prediction_angles = torch.zeros((num_pred,sin_orientation.shape[1]), device=predictions.device)

        # for every predicted center point find its center
        for i, pred in enumerate(center_pred):
            batch = int(pred[0])
            x,y = pred[1:3]

            score = pred[-1]

            if self.use_orientation_confidence_score:                
                confidence_score = prediction_confidence_score[batch,0, int(y), int(x)]                
                orientation_confidence_score[i,:] = confidence_score
                
                score *= confidence_score
            
            if score <= 0.9:
                continue

            s = sin_orientation[batch,:, int(y), int(x)]
            c = cos_orientation[batch,:, int(y), int(x)]
            
            pred_angle = torch.atan2(c, s)

            pred_angle = torch.rad2deg(pred_angle)
            pred_angle += 360 * (pred_angle < 0).int()

            prediction_angles[i,:] = pred_angle

        if self.use_orientation_confidence_score:
            center_pred = torch.cat((center_pred, orientation_confidence_score.to(predictions.device)),axis=1)

        # add orientations to list of returned values
        center_pred_with_rot = torch.cat((center_pred, prediction_angles.to(predictions.device)),axis=1)
        
        end_orient = time.time()
        times = list(times)
        times[-1] = times[-1] + (end_orient-start_orient)

        return center_pred_with_rot, tuple(times)