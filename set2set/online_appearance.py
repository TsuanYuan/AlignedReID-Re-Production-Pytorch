"""
online update of appearance model
Quan Yuan
2018-09-17
"""
import collections
import numpy
import os
import glob

class OnlineAppearanceModel(object):

    def __init__(self, num_feature_per_pid=16, top_ratio=0.1):
        """
        :param num_feature_per_pid: num of features to sample of each pid
        :param top_ratio: k-th of sorted crop-wise distances as a set distance, k=num_feature_per_pid^2*top_ratio
        """
        self.num_feature_per_pid = num_feature_per_pid
        self.pid_appearances = collections.defaultdict(list)
        self.top_ratio = top_ratio

    @staticmethod
    def sample_sparse_features(features, num):
        """
        :param features: the input list of features
        :param num: num of features in the sparse set
        :return: sparse set of features
        """
        feature_array = numpy.array(features)
        dist_m = 1- numpy.dot(feature_array, feature_array.transpose())
        # sort the distance and pick those with large distance to others
        mean_dist_to_others = numpy.mean(dist_m, axis=1)  # still includes 0 dist to itself, but anyway
        sorted_ids = numpy.argsort(mean_dist_to_others)
        max_dist_ids = sorted_ids[-num:]   # items with largest distances to others
        sampled_features = feature_array[max_dist_ids,:]
        return sampled_features.tolist()

    def add_features_of_pid(self, features, pid):
        """
        :param features: ReID features from the tracklets of this pid
        :param pid: the pid
        :return: None
        """

        if len(self.pid_appearances[pid]) + len(features) < self.num_feature_per_pid:
            self.pid_appearances[pid] += features
        else:
            features_combined = self.pid_appearances[pid] + features
            self.pid_appearances[pid] = self.sample_sparse_features(features_combined, self.num_feature_per_pid)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum(axis=0)

    def compute_class_probabilities(self, new_feature):
        """
        compute the probabilities of a new feature that belongs to any of existing pids
        :param new_feature: input new feature vector
        :return: a vector of soft_max_scores to each pid, a dict of probabilities to each pid
        """
        feature_len = len(numpy.squeeze(new_feature))
        pid_dists = {}
        for pid in self.pid_appearances:
            pid_features = numpy.array(self.pid_appearances[pid])
            dist = numpy.squeeze(1-numpy.dot(new_feature.reshape((1, feature_len)), pid_features.transpose()))
            pick_id = max(1, min(dist.size-1, int(round(self.top_ratio*dist.size))))
            pid_dists[pid] = numpy.sort(dist)[pick_id]

        dist_array = numpy.array([v for k, v in pid_dists.iteritems()])
        soft_max_scores = self.softmax(1-dist_array)
        return soft_max_scores, pid_dists


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_folder", type=str, help="path to input folder of tracklets")
    ap.add_argument("ext", type=str, help="extension to load feature")
    ap.add_argument("--count_per_id", type=int, help="count of training pids", default=16)
    ap.add_argument("--top_ratio", type=float, help="count of testing pids", default=0.1)
    args = ap.parse_args()

    print "count per ID = {}, top k distance ratio = {}".format(str(args.count_per_id), str(args.top_ratio))

    sub_folders = next(os.walk(args.data_folder))[1]  # [x[0] for x in os.walk(folder)]
    tps = []
    model = OnlineAppearanceModel(num_feature_per_pid=args.count_per_id, top_ratio=args.top_ratio)
    test_feature = {}
    for sub_folder in sub_folders:
        if sub_folder.isdigit():
            pid = int(sub_folder)
            sub_folder_full = os.path.join(args.data_folder, sub_folder)
            feature_files = glob.glob(os.path.join(sub_folder_full, '*.'+args.ext))
            features = [numpy.fromfile(f, dtype=numpy.float32) for f in feature_files]
            # adding features to the pid to construct a pid class for nearest neighbor search
            model.add_features_of_pid(features[:-1], pid)
            test_feature[pid] = features[-1]
    top1, top5 = 0, 0
    for pid in test_feature:
        soft_max_score, pid_dist = model.compute_class_probabilities(test_feature[pid])
        pid_list = numpy.array(pid_dist.keys())
        dist_list = numpy.array(pid_dist.values())
        sort_ids = numpy.argsort(dist_list)
        if pid_list[sort_ids[0]] == pid:
            top1+=1
        if pid in pid_list[sort_ids[0:5]].tolist():
            top5+=1
    n = len(test_feature)
    top1/=float(n)
    top5/=float(n)

    print "top 1 = {}, top 5 ={} on {} ids".format(str(top1), str(top5), str(n))
