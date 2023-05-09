#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from textwrap import dedent
from math import ceil

#Imports either a standard file OR uses a default standard file
try:
    import standard as std
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

    specific_style_greyscale = {
        'io_uring both': { 'lw': 1.2, 'ls': '-', 'color': (136/255, 34/255, 85/255)}, #Purple
        'io_uring spoll': { 'lw': 1.2, 'ls': '--', 'color': (136/255, 34/255, 85/255)}, #Purple
        'io_uring sqpoll': { 'lw': 1.2, 'ls': '-', 'color': (136/255, 34/255, 85/255)}, #Purple
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

    ioengine_rgb_colors = {
        'spdk': (136/255, 204/255, 238/255), #Cyan
        'io_uring': (68/255, 170/255, 153/255), #Teal
        'pvsync2': (51/255, 34/255, 136/255), #Indigo
        'libaio': (221/255, 204/255, 119/255), #Sand
        'posixaio': (136/255, 34/255, 85/255), #Wine
    }
    
    def get_style(name, gscale=False):
        if name in specific_style:
            from copy import copy
            if gscale:
                return copy(specific_style_greyscale[name])
            else:
                copy(specific_style[name])

        color = ioengine_color[None]
        for ioengine in ioengine_color:
            if ioengine and ioengine in name:
                if gscale:
                    color = ioengine_rgb_colors[ioengine]
                else:
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

    def convert_greyscale(r, g, b):
        a = float(0.299*r) + float(0.587*g) + float(0.114*b)
        return str(round(a, 4))

###############################################################################################################

#Formats y-axis ticks; ex. it turns '400000 into 400K'
def my_format(x, pos):
    if str(x) == '0.0':
        return str(int(x))
    elif x < 1000:
        return str(x)    
    elif (x/1000) % 1 == 0:
        return str(int(x/1000)) + 'K'
    else:
        return str(x/1000) + 'K'

###############################################################################################################

#Parses the dataframe
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

###############################################################################################################

parser = argparse.ArgumentParser(description=
                                 dedent('''
                                 CSL double figure graphing script
                                 '''))

parser.add_argument('data_file',
                    type=str,
                    help='CSV file to plot')

parser.add_argument('-p1', '--preset1',
                    type=str,
                    default='iops',
                    choices=['cpu_util', 'iops', 'watts_mean', 'iopj'],
                    help='Premade plot (1st Plot)')

parser.add_argument('-p2', '--preset2',
                    type=str,
                    default='iopj',
                    choices=['cpu_util', 'iops', 'watts_mean', 'iopj'],                    
                    help='Premade plots (2nd Plot)')

parser.add_argument('-px1',
                    type=str,
                    default='numjobs',
                    choices=['numjobs', 'iodepth'],
                    help='What to plot on preset\'s x-axis (1st Plot)')

parser.add_argument('-px2',
                    type=str,
                    default='numjobs',
                    choices=['numjobs', 'iodepth'],
                    help='What to plot on preset\'s x-axis (2nd Plot)')

parser.add_argument('-x1',
                    type=str,
                    help='**Manual Plotting** x-axis (1st Plot)')

parser.add_argument('-y1',
                    type=str,
                    help='**Manual Plotting** y-axis (1st Plot)')

parser.add_argument('-x2',
                    type=str,
                    help='**Manual Plotting** x-axis (2nd Plot)')

parser.add_argument('-y2',
                    type=str,
                    help='**Manual Plotting** y-axis (2nd Plot)')

parser.add_argument('-g', '--grey', '--gray',
                    type=bool,
                    nargs='?',
                    default=False,
                    const=True,
                    help='Converts and plots as greyscale')

parser.add_argument('--figsize',
                    type=int,
                    nargs=2,
                    default=(8, 4),
                    help='Figure size (w, h)')

parser.add_argument('--bs1',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size')

parser.add_argument('--bs2',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size')

parser.add_argument('-c', '--columns',
                    default=False,
                    const=True,
                    action='store_const',
                    help='Print out unique post-parsed columns of the DataFrame and do nothing else')

parser.add_argument('-d', '--debug',
                    default=False,
                    const=True,
                    action='store_const',
                    help='Enter debugging after parsing the DataFrame')

parser.add_argument('-o', '--output',
                    type=str,
                    help='Save as <path + filename>')

#Parse CLI Arguments
args = parser.parse_args()
args.px1 += '_parsed'
args.px2 += '_parsed'

#Read and parse the dataset
df = pd.read_csv('./data.csv')
df = parse_name_col(df)

#Printing columns
if args.columns:
    for x in np.unique(df.columns.values):
        print(x)
    exit()
elif args.debug:
    breakpoint()

#Filter by blocksize
dfs = [df[df.bs == args.bs1], df[df.bs == args.bs2]]

xlabel = [None] * 2
ylabel = [None] * 2

#Safety Checks
preset1 = True
preset2 = True
if bool(args.x1) ^ bool(args.y1):
    raise Exception('Either use both -x1 and -y1 or neither')
elif bool(args.x2) ^ bool(args.y2):
    raise Exception('Either use both -x2 and -y2 or neither')
else:
    if args.x1 and args.y1:
        if args.x1 not in dfs[0].columns.values:
            raise Exception('Chosen \'-x1\' value is not a column in the supplied dataset.')
        elif args.y1 not in dfs[0].columns.values:
            raise Exception('Chosen \'-y1\' value is not a column in the supplied dataset.')

        xlabel[0], ylabel[0] = args.x1, args.y1
        if xlabel[0] == 'numjobs' or xlabel[0] == 'iodepth':  xlabel[0] += '_parsed'
        if ylabel[0] == 'numjobs' or ylabel[0] == 'iodepth':  ylabel[0] += '_parsed'        
        preset1 = False
    if args.x2 and args.y2:
        if args.x2 not in dfs[1].columns.values:
            raise Exception('Chosen \'-x2\' value is not a column in the supplied dataset.')
        elif args.y2 not in dfs[1].columns.values:
            raise Exception('Chosen \'-y2\' value is not a column in the supplied dataset.')

        xlabel[1], ylabel[1] = args.x2, args.y2
        if xlabel[1] == 'numjobs' or xlabel[1] == 'iodepth':  xlabel[1] += '_parsed'
        if ylabel[1] == 'numjobs' or ylabel[1] == 'iodepth':  ylabel[1] += '_parsed'        
        preset2 = False
    
    if preset1:
        xlabel[0], ylabel[0] = args.px1, args.preset1
    if preset2:
        xlabel[1], ylabel[1] = args.px2, args.preset2

def no_unique_values(col):
    tmp = col.to_numpy()
    return (tmp[0] == tmp).all()

tmp_col1 = dfs[0][dfs[0]['ioengine'] == np.unique(dfs[0]['ioengine'])[0]][xlabel[0]]
tmp_col2 = dfs[1][dfs[1]['ioengine'] == np.unique(dfs[1]['ioengine'])[0]][xlabel[1]]

#More Safety Checks
if len(tmp_col1) < 2:
    raise Exception('The chosen \'-x1\' has less than 2 values for an ioengine in the DataFrame')
elif len(tmp_col2) < 2:
    raise Exception('The chosen \'-x2\' has less than 2 values for an ioengine in the DataFrame')
elif no_unique_values(tmp_col1):
    raise Exception('The chosen \'-x1\' has no unique values for an ioengine in the DataFrame')
elif no_unique_values(tmp_col2):
    raise Exception('The chosen \'-x2\' has no unique values for an ioengine in the DataFrame')
    
#Beginning of the graph setup
fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize))

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].yaxis.set_major_formatter(FuncFormatter(my_format))
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].yaxis.set_major_formatter(FuncFormatter(my_format))
Markers = {'spdk cpoll':'$♦$', 'pvsync2 cpoll':'$■$', 'libaio int':'$+$', 'pvsync2 int':'$□$', 'io_uring int':'$△$', 'io_uring cpoll':'$▲$', 'io_uring spoll':'$⃝$', 'io_uring both':'o', 'posixaio int':'x'}

try:
    for x in range(2):
        axes[x].set_xlabel(std.get_feature(xlabel[x]))
        axes[x].set_ylabel(std.get_feature(ylabel[x]))
except:
    for x in range(2):
        axes[x].set_xlabel(get_feature(xlabel[x]))
        axes[x].set_ylabel(get_feature(ylabel[x]))
        
try:
    axes[0].set_xscale('log', basex=2)
    axes[1].set_xscale('log', basex=2)
except:
    axes[0].set_xscale('log', base=2)
    axes[1].set_xscale('log', base=2)    

#Plotting

for x in range(2):
    for ioengine, group in dfs[x].groupby('ioengine'):
        f = group.sort_values(xlabel[x])
        try:
            style_grey = std.get_style(ioengine, args.grey)
            if args.grey:
                style_grey['color'] = std.convert_greyscale(*style_grey.get('color'))
            
            axes[x].plot(f[xlabel[x]], f[ylabel[x]], marker=Markers.get(ioengine), markersize=4, label=std.get_label(ioengine), **std.get_style(ioengine, args.grey))
        except:
            style_grey = get_style(ioengine, args.grey)
            if args.grey:
                style_grey['color'] = convert_greyscale(*style_grey.get('color'))

            axes[x].plot(f[xlabel[x]], f[ylabel[x]], marker=Markers.get(ioengine), markersize=4, label=get_label(ioengine), **style_grey)
        
        if(ylabel[x] == 'watts_mean'):
            axes[x].set_ylim(40, dfs[x][ylabel[x]].max() * 1.05)
        else:
            axes[x].set_ylim(0, dfs[x][ylabel[x]].max() * 1.05)

#Adding final graph settings
axes[0].grid(axis='both', ls='--', alpha=0.2)
axes[1].grid(axis='both', ls='--', alpha=0.2)
X_ticks1 = np.sort(dfs[0][xlabel[0]].unique())
X_ticks2 = np.sort(dfs[1][xlabel[1]].unique())
axes[0].xaxis.set_major_formatter(ScalarFormatter())
axes[1].xaxis.set_major_formatter(ScalarFormatter())
#ax.spines.bottom.set_bounds(df[x_label].min(), df[x_label].max())
axes[0].set_xticks(X_ticks1)
axes[1].set_xticks(X_ticks2)

fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, -0.12), labelspacing=0.15, handletextpad=0.15, frameon=False, fancybox=False, shadow=False, ncol=ceil(len(np.unique(df['ioengine'])) / 2)) #THIS WILL EVENTUALLY BE A PROBLEM DUE TO XLABEL

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

if args.output:
    plt.savefig(args.output, bbox_inches='tight', transparent=True)
else:
    print('\n\n***LEGEND WILL NOT SHOW CORRECTLY UNLESS SAVED***\n\n')
    plt.show()
