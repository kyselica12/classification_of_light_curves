from datetime import datetime
import glob
import os
from collections import namedtuple, defaultdict
from typing import List

import numpy as np
import pandas as pd
import tqdm
import logging

logging.basicConfig(filename="LOG.log")

class Preprocessing:
    Track = namedtuple("Track", "light_curve start period")

    def __init__(self, array_size=1000, len_threshold=0.5, window_size=5):
        self.array_size = array_size
        self.len_threshold = len_threshold
        self.window_size = window_size

        self.logger = logging.getLogger("Preprocessing")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"preprocessing innit -> array_size: {array_size}, len_threshold: {len_threshold}")

    def run(self, data_folder, output_folder=None):
        self.logger.info(f"run method -> input: {data_folder}, output: {output_folder}.")

        files = list(glob.iglob(f"{data_folder}/periods/*.txt"))
        for period_file in tqdm.tqdm(files, desc="Total progess: ", leave=None):
            if not self._good_file(period_file):
                continue

            ID, name, df = self.load_period_file(period_file)

            if df.empty:
                self.logger.info(f"File: {os.path.split(period_file)[1]} empty file.")
                continue

            tracks = self.load_light_curves(data_folder, ID, df)

            if len(tracks) == 0:
                continue

            light_curves = self.curves_from_tracks(tracks, self.array_size)
            light_curves = self._filter_curves(light_curves)

            if output_folder:
                self.save_to_file(light_curves, name, output_folder)


    def save_to_file(self, light_curves, name, output_folder):
        file_name = f"{output_folder}/{name}.npy"
        with open(file_name, "ab") as csv:
            np.save(csv, light_curves)

    def curves_from_tracks(self, tracks, size):

        data = np.zeros((len(tracks), size))
        for i in tqdm.tqdm(range(len(tracks)), desc="Light curves from tracks: ", leave=None):
            track = tracks[i]
            data[i] = self.resize_curve(track.light_curve, size)

        return data

    def resize_curve(self, light_curve, size):

        data = np.zeros(size)

        j = 0
        step = 1 / size

        for i in range(size):
            t = step * i
            s = t - 0.5 * step

            k = j
            points: List[float] = []
            while k < len(light_curve):
                observation_time = light_curve[k][1]
                if observation_time < t + 0.5 * step:
                    points.append(light_curve[k][0])
                if observation_time < t:
                    j += 1
                k += 1
            points = np.array(points, dtype=np.float32)
            if len(points) > 0:
                data[i] = np.mean(points)

        return data

    def SMA(self, light_curve, target_size):
        N = len(light_curve)
        step = 1 / (target_size + self.window_size)

        data = np.zeros(target_size)
        window_t = step * self.window_size

        t = window_t
        idx1, idx2 = 0, 0

        for i in range(target_size):

            if idx1 == len(light_curve):
                break

            while idx2 < N and light_curve[idx2][1] <= t:
                idx2 += 1
            while idx1 < N and light_curve[idx1][1] < t - window_t:
                idx1 += 1

            window_data = light_curve[idx1:idx2, 0]
            if len(window_data):
                data[i] = np.mean(window_data)
            t += step

        return data

    def _to_timestamp(self, date, time):
        date_time = f'{date} {time}'
        sec = datetime.strptime(date_time, f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}").timestamp()
        ns = int(sec * 10 ** 9)
        return ns

    def load_period_file(self, path):
        df = None
        ID = int(os.path.split(path)[1][:-len('_periods.txt')])

        with open(path, 'r') as period_file:
            lines = period_file.readlines()

        data = []
        name = self._get_object_name_from_first_line(lines[0])

        for line in filter(lambda l: not l.startswith("#"), lines):
            date, time, track, mjd, period = line.split()
            data.append([self._to_timestamp(date, time),
                         int(track),
                         float(mjd),
                         int(float(period) * 10 ** 9)])  # period from sec to ns

        df = pd.DataFrame(data,
                          columns=["Timestamp", "Track", "MJD", "Period"])

        return ID, name, df

    def _get_object_name_from_first_line(self, line):
        return line.split()[3].replace('/', '_')

    def _good_file(self, path):
        if not os.path.isfile(path):
            self.logger.info(f"File: {os.path.split(path)[1]} does not exist.")
            return False

        with open(path, 'r') as f:
            line = f.readline()
        if not line.startswith("#"):
            self.logger.info(f"File: {os.path.split(path)[1]} bad request.")
            return False

        return True

    def load_object_file(self, path):

        with open(path, 'r') as object_file:
            lines = object_file.readlines()

        data = []

        # Request Error -> contains HTML
        if not lines[0].startswith("#"):
            return None
        file = os.path.split(path)[1]
        for line in tqdm.tqdm(lines, desc=f"Load object file {file}: ", leave=None):
            if line.startswith("#"):
                continue

            date, time, stMag, mag, filter, penumbra, distance, phase, channel, track = line.split()

            data.append([int(track),
                         self._to_timestamp(date, time),
                         float(stMag),
                         float(mag),
                         filter,
                         float(penumbra),
                         float(distance),
                         float(phase),
                         int(channel)])

        df = pd.DataFrame(data,
                          columns=["Track", "Timestamp", "StdMag",
                                   "Mag", "Filter", "Penumbra", "Distance",
                                   "Phase", "Channel"])

        return df

    def load_light_curves(self, data_folder, ID, df_periods):
        object_file_name = f"{data_folder}/objects/{ID}_tracks.txt"
        tracks = []

        # non existing file or bad file
        if not self._good_file(object_file_name):
            return []

        df_tracks = self.load_object_file(object_file_name)

        # empty file
        if df_tracks.empty:
            self.logger.info(f"File: {os.path.split(object_file_name)[1]} empty file.")
            return []

        for _, track in tqdm.tqdm(df_periods.iterrows(), total=df_periods.shape[0], desc="Load light curves: ", leave=None):
            track_n = track["Track"]

            measurements = df_tracks[df_tracks["Track"] == track_n]
            if len(measurements) == 0:
                continue

            time_period = track["Period"]

            tracks.extend(self.create_tracks_by_period(measurements, time_period))

        return tracks

    def create_tracks_by_period(self, measurements, time_period):
        light_curves = []

        measurements = measurements.sort_values(by="Timestamp")

        start = measurements.iloc[0]["Timestamp"]
        end = start + time_period

        curve = []
        for i, m in measurements.iterrows():
            t = m["Timestamp"]
            while t > end:
                if curve != []:
                    light_curves.append(Preprocessing.Track(np.array(curve), start, time_period))
                    curve = []
                start = end
                end = end + time_period

            curve.append([m["StdMag"], (m["Timestamp"] - start) / time_period])

        if curve != []:
            light_curves.append(Preprocessing.Track(np.array(curve), start, time_period))

        return light_curves

    def _filter_curves(self, curves):
        '''Filter out short light curves'''
        return curves[np.sum(curves != 0, axis=1)/self.array_size > self.len_threshold]

if __name__ == "__main__":
    PATH = "c:/Users/danok/Documents/charon_share/data"  # "/home/daniel/Desktop/charon_share/test_data"

    p = Preprocessing(1000)

    p.run(PATH, "../data")
