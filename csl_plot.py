#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from textwrap import dedent

try:
    import standard
except:
    ioengine_color = {
        'spdk': '#88CCEE', #Cyan
        'io_uring': '#44AA99', #Teal
        'pvsync2': '#332288', #Indigo
        'libaio': '#DDCC77', #Sand
        'posixaio': '#882255', #Wine
        None: 'k',
    }
    
    completion_ls = {
        'int': '--',
        'poll': '-',
        'hipri': '-',
        'cpoll': '-',
        'sqpoll': '..',
        'spoll': '..',
        'both': '-.',
        None: '-',
    }
    
    specific_style = {
        'io_uring both': { 'lw': 1.2, 'ls': '-', 'color': '#CC6677'}, #Purple
        'io_uring spoll': { 'lw': 1.2, 'ls': '--', 'color': '#CC6677'}, #Purple
        'io_uring sqpoll': { 'lw': 1.2, 'ls': '-', 'color': '#CC6677'}, #Purple
    }
    
    feature_remap = {
        #'read_lat_mean': 'Read latency (\u03BCs)',
        'read_lat_mean': 'Latency (\u03BCs)',
        #'iodepth': 'IO depth (number of threads)',
        'iodepth': 'Number of Tenants',
        'numjobs': 'IO Depth',
        'numjobs_parsed': 'Number of Tenants',
        'iodepth_parsed': 'IO Depth',
        'bs': 'Request size',
        'iops': 'IOPS',
        'iopj': 'IOPJ',
        #'read_iops': 'Read IOPS',
        'read_iops': 'IOPS',
        'read_bandwidth': 'Throughput (MB/s)',
        'watts_mean': 'Power',
        'watts_mean_std': '\u03c3 of Power (W)',
        'loadstore_per_sec': '(Loads + Stores) per sec.',
        'memory_bound_pc': 'Memory bound time (%)',
        'Observed_Maximum': 'Peak DRAM bandwidth (GB/s)',
        'cpu_csw': 'Context switches',
        'csw_per_sec': 'Context switches per sec.',
        'csw_per_io': 'Context switches per IO',
        'cpu_sys': 'System CPU',
        'cpu_user': 'User CPU',
        'cpu_util': 'CPU (user + system)',
        'cpu_total': 'Total CPU',
        'poll_invoked_per_sec': 'Polls per second',
        'poll_invoked_per_io': 'Polls per IO',
        'loadstore_per_io': '(Loads + Stores) per IO',
        'bytes_per_joule': 'Bytes read per joule',
        'threads': 'Number of threads'
    }
    
    legend_remap = {
        'io_uring both': 'io_uring (sp, cp)',
        'io_uring cpoll': 'io_uring (syscall, cp)',
        'io_uring spoll': 'io_uring (sp, int)',
        'io_uring int': 'io_uring (syscall, int)',
        'libaio int': 'libaio',
        'pvsync2 int': 'posix-sio (int)',
        'pvsync2 cpoll': 'posix-sio (cp)',
        'posixaio int': 'posix-aio',
        'spdk cpoll': 'spdk',
    }

    feature_unit = {
        'watts_mean': 'W',
        'loadstore_per_sec': '/s',
        'bytes_per_joule': 'B/J',
        'csw_per_sec': '/s',
        'read_bandwidth': 'B/s',
        #'iopj': 'Energy Efficiency',
        'cpu_total': 'cores',
    }
    
    def get_style(name):
        if name in specific_style:
            return specific_style[name]

        color = ioengine_color[None]
        for ioengine in ioengine_color:
            if ioengine and ioengine in name:
                color = ioengine_color[ioengine]
                break

        ls = completion_ls[None]
        for ls in completion_ls:
            if ls and ls in name:
                ls = completion_ls[ls]
                break
        return { 'lw': 1.2, 'ls': ls, 'color': color, }

    def get_feature_unit(feature):
        return feature_unit.get(feature, '')

    def get_feature(feature):
        s = feature_remap.get(feature, feature)
        if feature in feature_unit:
            s += ' (' + feature_unit[feature] + ')'
        return s

    def get_label(feature):
        return legend_remap.get(feature, feature)

def parse_name_col(df):
    """
    Parse the name column to extract more fields
    """
    parsed_name = df.name.str.extract(r'^(?P<completion>classic_q1|interrupt)_(?P<sched>[^_]+)_(?P<ioengine>io_uring|[^_]+)_(?P<rw>[^_]+)-(?P<bs>\d+K)_iodepth(?P<iodepth>\d+)_numjobs(?P<numjobs>\d+)_submit(?P<submit_size>\d+)_complete(?P<completion_size>\d+)(?P<cpoll>_hipri)?(?P<spoll>_sqthread)?')

    for col in parsed_name.columns:
        try:
            parsed_name[col] = parsed_name[col].astype(int)
        except ValueError:
            try:
                parsed_name[col] = parsed_name[col].astype(float)
            except:
                pass
    parsed_name.fillna(False, inplace=True)
    parsed_name['cpoll'].replace({'_hipri': True}, inplace=True)
    parsed_name['spoll'].replace({'_sqthread': True}, inplace=True)
    parsed_name['both'] = parsed_name['cpoll'] & parsed_name['spoll']

    both = parsed_name['both']
    parsed_name.loc[both, 'ioengine'] += ' both'
    
    cpoll = parsed_name['cpoll'] & ~parsed_name['spoll']
    parsed_name.loc[cpoll, 'ioengine'] += ' cpoll'
    
    spoll = parsed_name['spoll'] & ~parsed_name['cpoll']    
    parsed_name.loc[spoll, 'ioengine'] += ' spoll'

    ints = ~parsed_name['cpoll'] & ~parsed_name['spoll']
    parsed_name.loc[ints & (parsed_name['ioengine'].str.contains("spdk") != True), 'ioengine'] += ' int'

    parsed_name.loc[parsed_name['ioengine'].str.contains('spdk') == True, 'ioengine'] += ' cpoll'

    return df.join(parsed_name, rsuffix='_parsed')

parser = argparse.ArgumentParser(description=
                                 dedent('''
                                 TEST

                                 '''))

parser.add_argument('data_file',
                    type=str,
                    help='CSV file to plot')

parser.add_argument('-p', '--preset',
                    type=str,
                    default='iops',
                    choices=['cpu_util', 'iops', 'watts_mean', 'iopj'],
                    help='Premade plots')

parser.add_argument('-px',
                    type=str,
                    default='numjobs',
                    choices=['numjobs', 'ioengine'],
                    help='What to plot on preset\'s x-axis')

parser.add_argument('-x',
                    type=str,
                    help='**Manual Plotting** x-axis')

parser.add_argument('-y',
                    type=str,
                    help='**Manual Plotting** y-axis')

parser.add_argument('--fs',
                    type=tuple,
                    default=(8, 4),
                    help='Figure size ( , )')

parser.add_argument('--bs',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size')

parser.add_argument('-o', '--output',
                    type=str,
                    help='Path to output file')

args = parser.parse_args()
args.px += '_parsed'

df = pd.read_csv('./data.csv')
df = parse_name_col(df)

if bool(args.x) ^ bool(args.y):
    raise Exception('Either use both -x and -y or neither')
else:
    if args.x and args.y:
        x_label, y_label = args.x, args.y
    elif args.preset:
        x_label, y_label = args.px, args.preset
    else:
        raise Exception('You broke something!')

breakpoint()
