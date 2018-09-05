"""
train tracklet model
Quan Yuan
2018-09-04
"""
import argparse
import pickle



def train(args):
    for epoch in range(args.num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            pass

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('tracklet_data', type=str,
                        help='the path to tracker result')

    parser.add_argument('model_path', type=str,
                        help='the path to output the model')

    parser.add_argument('--id_type', type=str, default='long_tracklet_id',
                        help='type of person ids, like pid, long_tracklet_id, tracklet_index')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    args = parser.parse_args()

    with open(args.output_file, 'rb') as fp:
        data = pickle.load(fp)
