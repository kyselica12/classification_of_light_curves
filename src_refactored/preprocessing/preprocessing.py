import time
from datetime import datetime
import glob
import os
from collections import namedtuple, defaultdict
from typing import List

import numpy as np
import pandas as pd
import tqdm
import logging


class Preprocessing:
    Track = namedtuple("Track", "light_curve start period")

    def __init__(self, array_size=1000, len_threshold=0., window_size=1):
        self.array_size = array_size
        self.len_threshold = len_threshold
        self.window_size = window_size

        self.logger = logging.getLogger("Preprocessing",)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"preprocessing innit -> array_size: {array_size}, len_threshold: {len_threshold}, "
                         f"window_size: {window_size}")

    def run(self, data_folder, output_folder=None, from_idx=0, num=None):
        """
        Creates numpy multi arrays in output folder containing light curves from input folder
        :param data_folder:
        :param output_folder:
        :return:
        """
        self.logger.info(f"run method -> input: {data_folder}, output: {output_folder}.")

        files = list(glob.iglob(f"{data_folder}/periods/*.txt"))
        files_from_index = filter(lambda x: self.get_ID(x) >= from_idx, files)
        files = sorted(files_from_index, key=lambda x: self.get_ID(x))

        if num:
            files = files[:num]

        for period_file in tqdm.tqdm(files, desc="Total progess: ", leave=None):
            try:
                if not self._good_file(period_file):
                    continue

                ID, name, df = self.load_period_file(period_file)

                if df.empty:
                    self.logger.info(f"File: {os.path.split(period_file)[1]} empty file.")
                    continue

                tracks = self.load_light_curves(data_folder, ID, df)

                if len(tracks) == 0:
                    continue

                light_curves = self.curves_from_tracks(tracks)
                light_curves = self._filter_curves(light_curves)

                if output_folder:
                    self.save_to_file(light_curves, name, output_folder)
            except Exception as e:
                self.logger.error(f"Error ocured in reading file: {period_file} {e}")

    def save_to_file(self, light_curves, name, output_folder):
        """
        Append light curves to file. Creates multi_array
        :param light_curves: numpy array
        :param name: name of the object
        :param output_folder: output folder
        :return:
        """
        name = name.replace(" ", "_").replace('/', '|')
        file_name = f"{output_folder}/{name}_multi_array.npy"
        with open(file_name, "ab") as f:
            np.save(f, light_curves)

    def curves_from_tracks(self, tracks):

        data = np.zeros((len(tracks), self.array_size))
        for i in tqdm.tqdm(range(len(tracks)), desc="Light curves from tracks: ", leave=None):
            track = tracks[i]
            data[i] = self.simple_moving_average(track.light_curve)

        return data

    def simple_moving_average(self, light_curve):
        """
        Apply simple moving average to resize light curve to self.array_size
        :param light_curve:
        :return: np.array light curve
        """
        N = len(light_curve)
        step = 1 / (self.array_size + self.window_size)

        data = np.zeros(self.array_size)
        window_t = step * self.window_size

        t = window_t
        idx1, idx2 = 0, 0

        for i in range(self.array_size):

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
        """
        Loads period files - contains information about tracks of one object
        :param path: period folder
        :return: number of period file, name of the object, dataframe
        """
        df = None
        ID = self.get_ID(path)

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

    def get_ID(self, path):
        return int(os.path.split(path)[1][:-len('_periods.txt')])

    def _get_object_name_from_first_line(self, line):
        name = ' '.join(line.split()[3:]) # only take inforamtion after norad number
        name = name.split(' / ')[0] # remove country of origin
        return name

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
        """
        Loads tracks_{ID}.txt files containing measurements.
        :param path: input objects folder
        :return: dataframe
        """

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
        """
        Loads measurements for information from period file
        :param data_folder:
        :param ID: period file number
        :param df_periods: period information dataframe
        :return: List[track]
        """

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
        """
        Cut measurement by rotation period.
        :param measurements:
        :param time_period:
        :return: List of subcurves
        """
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