import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils

'''profiler_dir = "./step-tetris/profiler/1708884994"
directory = './step-tetris/'

p = utils.merge_profile_stats(profiler_dir)
utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime').print_stats()
p.strip_dirs().sort_stats('tottime').print_stats(20)

profiler_dir = "./step-tetris/profiler/1708885902"
directory = './step-tetris/'

p = utils.merge_profile_stats(profiler_dir)
utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime').print_stats()
p.strip_dirs().sort_stats('tottime').print_stats(20)'''

import glob

profiler_dir = "./step-tetris/profiler/"
dirs = glob.glob(profiler_dir + "*")
latest_prof_dir = max(dirs, key=lambda d: int(d.split("/")[-1]))

print(f"Using profiler data from {latest_prof_dir}")

directory = './step-tetris/'

p = utils.merge_profile_stats(latest_prof_dir)
utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime').print_stats()
p.strip_dirs().sort_stats('tottime').print_stats(20)