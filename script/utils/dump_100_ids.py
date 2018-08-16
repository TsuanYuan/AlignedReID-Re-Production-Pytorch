"""
dump tracklets from QiaoHui's collection of track IDs
Quan Yuan
2018-08-16
"""

from openpyxl import load_workbook
import argparse
import os


def load_xlsl(xlsl_file):
    wb = load_workbook(filename=xlsl_file)
    sheet1 = wb['Sheet1']
    pos = sorted(sheet1._cells.keys())
    video_track_table = {}
    i = 0
    for p in pos:
        if p[1] > 2:
            continue
        if i%2==0:
            video_name = str(sheet1._cells[p].value)
        else:
            track_id = int(sheet1._cells[p].value)
            video_base, _ = os.path.splitext(video_name)
            if video_base not in video_track_table:
                video_track_table[video_base] = []
            video_track_table[video_base].append(track_id)
        i+=1
    return video_track_table

def load_track_data(result_folder, video_track_table):
    pass

def process(video_folder, tracks):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('video_folder', type=str,
                        help='folder of videos')
    parser.add_argument('result_folder', type=str,
                        help='folder of videos')
    parser.add_argument('input_list_file', type=str,
                        help='input list')

    args = parser.parse_args()
    video_track_table = load_xlsl(args.input_list_file)
    tracks = load_track_data(args.result_folder, video_track_table)

    process(args.video_folder, tracks)
