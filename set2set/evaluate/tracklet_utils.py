"""
utils for tracklet to tracklet matching
Quan Yuan
2018-10-18
"""
import numpy
import load_model
import sklearn


class FeatureExtractor(object):
    def __init__(self, model_path, device_ids, sample_size=-1, mode='median', aggregate='ten_percent'):
        self.model = load_model.AppearanceModelForward(model_path=model_path, device_ids=device_ids)
        self.sample_size = sample_size
        self.mode = mode
        self.aggregate = aggregate

    def compute_features_on_batch(self, image_list):
        images = numpy.array(image_list)
        if self.sample_size > 0:
            sample_ids = numpy.round(numpy.linspace(0, len(images)-1, self.sample_size)).astype(int)
            images = images[sample_ids]

        images = self.model.normalize_images(images)
        features = self.model.compute_features_on_batch(images)
        return features

    def compute_single_feature(self, image_list):
        features = self.compute_features_on_batch(image_list)
        if self.mode == 'median':
            if features.size == 0:
                x = 0
            x = numpy.median(features, axis=0)
            feature = x / (numpy.linalg.norm(x)+0.0000001)
        elif self.mode == 'mean':
            x = numpy.mean(features, axis=0)
            feature = x / (numpy.linalg.norm(x) + 0.0000001)
        else:
            raise Exception('unknown mode of combine features')
        return feature

    def __call__(self, image_list):
        if self.aggregate == 'single':
            return self.compute_single_feature(image_list)
        elif self.aggregate.find('percent') >= 0:
            return self.compute_features_on_batch(image_list)


class TrackletComparison(object):
    def __init__(self, mode):
        self.mode = mode

    def compute_track_distance(self, tracks_1, tracks_2):
        if self.mode.find('single') >= 0:
            return 1 - numpy.dot(tracks_1, tracks_2)
        elif self.mode.find('ten_percent')>=0:
            dist_matrix = sklearn.metrics.pairwise_distances(tracks_1, tracks_2, metric="cosine")
            return numpy.percentile(dist_matrix, 10)
        else:
            print "unknown comparison option {}".format(self.mode)

    def __call__(self,track_1, track_2):
        return self.compute_track_distance(track_1, track_2)
