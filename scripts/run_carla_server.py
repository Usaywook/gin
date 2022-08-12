import subprocess
import os
import argparse

def main():
    cur_dir = os.getcwd()
    subprocess.call(cur_dir + '/kill_carla_server.sh')
    subprocess.call('/home/swyoo/usaywook/simulator/carla0911/CarlaUE4.sh')

if __name__ == '__main__':
    main()