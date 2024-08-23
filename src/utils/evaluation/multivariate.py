import copy
from functools import partial
class MultivariateEval:
    def __init__(self, eval_obj, image2tags_fn):

        # save a reference to the original eval object
        self.ref_eval_obj = copy.deepcopy(eval_obj)

        self.eval_obj = eval_obj
        self.image2tags_fn = image2tags_fn

        self.eval_obj_per_tag = dict()

    def add_image_prediction(self, im_name, *args, **kwargs):
        # call original evaluation class
        result = self.eval_obj.add_image_prediction(im_name, *args, **kwargs)

        # parse image name to get the object tags
        img_tags = self.image2tags_fn(im_name)

        # add metrics to the corresponding tag (call the sam function on the corresponding eval object)
        for t in img_tags:
            if t not in self.eval_obj_per_tag:
                # make a copy of the original eval object
                self.eval_obj_per_tag[t] = copy.deepcopy(self.ref_eval_obj)
                # add modify its save_str() function to add tag information
                self.eval_obj_per_tag[t].save_str = partial(lambda x:  self.eval_obj.save_str() + '_' + x, t)

            # this is less efficient since we call the same function multiple times but is independent of the eval class
            self.eval_obj_per_tag[t].add_image_prediction(im_name, *args, **kwargs)

        return result

    def save_str(self):
        return self.eval_obj.save_str()

    def get_attributes(self):
        return self.eval_obj.get_attributes()

    def get_results_timestamp(self, *args, **kwargs):
        return self.eval_obj.get_results_timestamp(*args, **kwargs)

    def calc_and_display_final_metrics(self, *args, **kwargs):
        # save original results
        metrics = self.eval_obj.calc_and_display_final_metrics(*args, **kwargs)

        # save per tag results
        for t, eval_obj in self.eval_obj_per_tag.items():
            self.eval_obj_per_tag[t].calc_and_display_final_metrics(*args, **kwargs)

        return metrics