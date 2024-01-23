import argparse
import logging
import time

from preprocessing import Preprocessing

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", '--input',
                        required=True,
                        type=str,
                        help="Path to directory concaining input objects folder and periods folder")

    parser.add_argument("-o", '--output',
                        required=False,
                        default='./result',
                        type=str,
                        help="Path to destination folder")

    parser.add_argument("-l", "--size",
                        required=True,
                        type=int,
                        help="Size of created light curves")

    parser.add_argument("-t", "--len-threshold",
                        type=float,
                        required=True,
                        help="Threshold between 0 and 1, if more than t * points in curve are missing, the curve is "
                             "discarded "
                        )
    parser.add_argument("-w", "--window-size",
                        required=False,
                        default=1,
                        type=int,
                        help="Length of window for simple moving average method")

    parser.add_argument("-s", '--start-idx',
                        required=False,
                        default=0,
                        type=int,
                        help="Starting object ID")
    parser.add_argument("-n", "--num",
                        required=False,
                        default=None,
                        type=int,
                        help="Number of objects to process.")


    args = parser.parse_args()

    logging.basicConfig(filename=f"../log/LOG_{int(time.time())}.log", level=logging.INFO)
    logging.info(f"Call: main.py -i {args.input} -o {args.output} -l {args.size} -t {args.len_threshold} -w {args.window_size}")

    p = Preprocessing(array_size=args.size, len_threshold=args.len_threshold, )
    p.run(args.input, args.output, from_idx=args.start_idx, num=args.num)