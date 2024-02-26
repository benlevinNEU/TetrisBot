import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils

from pstats import func_std_string

np = 12
sp = 3

def f8(x):
    return "%8.7f" % x

def print_title(p, gen=True):

    if gen:
        print('   ncalls    tottime/gen     percall    cumtime/gen      percall', end=' ', file=p.stream)

    else:
        print('   ncalls        tottime     percall        cumtime      percall', end=' ', file=p.stream)
    
    print('  filename:lineno(function)', file=p.stream)

def print_line(p, func, ng):  # hack: should print percentages

    cc, nc, tt, ct, callers = p.stats[func]
    c = str(nc)
    #if nc != cc:
    #    c = c + '/' + str(cc)
    print(c.rjust(np-3), end=' '*sp, file=p.stream)
    print(f8(tt/ng).rjust(np), end=' '*sp, file=p.stream)
    if nc == 0:
        print(' '*np, end=' '*sp, file=p.stream)
    else:
        print(f8(tt/nc).rjust(np-4), end=' '*sp, file=p.stream)
    print(f8(ct/ng).rjust(np), end=' '*sp, file=p.stream)
    if cc == 0:
        print(' '*np, end=' '*sp, file=p.stream)
    else:
        print(f8(ct/cc).rjust(np-2), end=' '*sp, file=p.stream)
    print(func_std_string(func), file=p.stream)

def print_stats(p, *amount):

    desired_key = None

    for key in p.stats.keys():
        if "evaluate_population" in key:
            desired_key = key
            break

    if desired_key is not None:
        ng = p.stats[desired_key][0]
        print(f"\nGenerations: {ng} in %.3f seconds" % p.total_tt, file=p.stream)

    width, list = p.get_print_list(amount)
    if list:
        print_title(p, desired_key is None)
        for func in list:
            print_line(p, func, ng)
        print(file=p.stream)
    return p

if __name__ == "__main__":
    import glob

    profiler_dir = "./frame-tetris/profiler/"
    dirs = glob.glob(profiler_dir + "*")
    latest_prof_dir = max(dirs, key=lambda d: int(d.split("/")[-1]))

    print(f"Using profiler data from {latest_prof_dir}")

    directory = './frame-tetris/'

    p = utils.merge_profile_stats(latest_prof_dir)
    print_stats(utils.filter_methods(p, directory).strip_dirs().sort_stats('tottime'))
    print_stats(p.strip_dirs().sort_stats('tottime'), 30)