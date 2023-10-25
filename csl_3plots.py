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
    Markers = {'spdk cpoll':'$♦$', 'pvsync2 cpoll':'$■$', 'libaio int':'$+$', 'pvsync2 int':'$□$', 'io_uring int':'$△$', 'io_uring cpoll':'$▲$', 'io_uring spoll':'$⃝$', 'io_uring both':'o', 'posixaio int':'x'}
    
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
        'iodepth': 'Number of Threads',
        'numjobs': 'IO Depth',
        'numjobs_parsed': 'Number of Threads',
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

    fs_ioengine_colors = {
        'xfs': 'C2',
        'ext4': 'C0',
        'f2fs': 'C1'
    }

    def get_fs_style(fs):
        color = fs_ioengine_colors.get(fs)
        return { 'lw': 1.2, 'ls': '-', 'color': color }
    
    def get_style(name, gscale=False):
        if name in specific_style:
            if gscale:
                return specific_style_greyscale.get(name)
            else:
                return specific_style.get(name)

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
        return { 'lw': 1.2, 'ls': ls, 'color': color}

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
        return '0'
    elif x < 1000:
        return str(int(x))
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

def no_unique_values(col):
    """
    Simple check if a DataFrame column has unique values
    """
    tmp = col.to_numpy()
    return (tmp[0] == tmp).all()

def safety_checks(dfs, args):
    """
    Performs script configuration safety checks.
    
    Is called by the 'filter_and_safety' function.
    """

    xlabels = [None] * 3
    ylabels = [None] * 3

    ### Safety Checks
    preset1 = True
    preset2 = True
    preset3 = True
    if bool(args.x1) ^ bool(args.y1):
        raise Exception('Either use both -x1 and -y1 or neither')
    elif bool(args.x2) ^ bool(args.y2):
        raise Exception('Either use both -x2 and -y2 or neither')
    elif bool(args.x3) ^ bool(args.y3):
        raise Exception('Either use both -x3 and -y3 or neither')
    else:
        if args.x1 and args.y1:
            if args.x1 not in dfs[0].columns.values:
                raise Exception('Chosen \'-x1\' value is not a column in the supplied dataset.')
            elif args.y1 not in dfs[0].columns.values:
                raise Exception('Chosen \'-y1\' value is not a column in the supplied dataset.')

            xlabels[0], ylabels[0] = args.x1, args.y1
            if xlabels[0] == 'numjobs' or xlabels[0] == 'iodepth':  xlabels[0] += '_parsed'
            if ylabels[0] == 'numjobs' or ylabels[0] == 'iodepth':  ylabels[0] += '_parsed'        
            preset1 = False
        
        if args.x2 and args.y2:
            if args.x2 not in dfs[1].columns.values:
                raise Exception('Chosen \'-x2\' value is not a column in the supplied dataset.')
            elif args.y2 not in dfs[1].columns.values:
                raise Exception('Chosen \'-y2\' value is not a column in the supplied dataset.')

            xlabels[1], ylabels[1] = args.x2, args.y2
            if xlabels[1] == 'numjobs' or xlabels[1] == 'iodepth':  xlabels[1] += '_parsed'
            if ylabels[1] == 'numjobs' or ylabels[1] == 'iodepth':  ylabels[1] += '_parsed'        
            preset2 = False

        if args.x3 and args.y3:
            if args.x3 not in dfs[2].columns.values:
                raise Exception('Chosen \'-x3\' value is not a column in the supplied dataset.')
            elif args.y3 not in dfs[2].columns.values:
                raise Exception('Chosen \'-y3\' value is not a column in the supplied dataset.')

            xlabels[2], ylabels[2] = args.x3, args.y3
            if xlabels[2] == 'numjobs' or xlabels[2] == 'iodepth':  xlabels[2] += '_parsed'
            if ylabels[2] == 'numjobs' or ylabels[2] == 'iodepth':  ylabels[2] += '_parsed'        
            preset3 = False
    
        if preset1:
            xlabels[0], ylabels[0] = args.px1, args.preset1
        if preset2:
            xlabels[1], ylabels[1] = args.px2, args.preset2
        if preset3:
            xlabels[2], ylabels[2] = args.px3, args.preset3

    tmp_col1 = dfs[0][dfs[0]['ioengine'] == np.unique(dfs[0]['ioengine'])[0]][xlabels[0]]
    tmp_col2 = dfs[1][dfs[1]['ioengine'] == np.unique(dfs[1]['ioengine'])[0]][xlabels[1]]
    tmp_col3 = dfs[2][dfs[2]['ioengine'] == np.unique(dfs[2]['ioengine'])[0]][xlabels[2]]

    ### More Safety Checks
    if len(tmp_col1) < 2:
        raise Exception('The chosen \'-x1\' has less than 2 values for an ioengine in the DataFrame')
    elif len(tmp_col2) < 2:
        raise Exception('The chosen \'-x2\' has less than 2 values for an ioengine in the DataFrame')
    elif len(tmp_col3) < 3:
        raise Exception('The chosen \'-x3\' has less than 2 values for an ioengine in the DataFrame')
    elif no_unique_values(tmp_col1):
        raise Exception('The chosen \'-x1\' has no unique values for an ioengine in the DataFrame')
    elif no_unique_values(tmp_col2):
        raise Exception('The chosen \'-x2\' has no unique values for an ioengine in the DataFrame')
    elif no_unique_values(tmp_col3):
        raise Exception('The chosen \'-x3\' has no unique values for an ioengine in the DataFrame')
    
    return dfs, xlabels, ylabels

def filter_and_safety(df, args):
    """
    Filters data based of options and runs 'safety_checks' function
    """
    
    ### Filter by bs and fs
    dfs = []
    for bs, fs in [(args.bs1, args.fs1), (args.bs2, args.fs2), (args.bs3, args.fs3)]:
        if fs != 'all':
            filtered_indices = [list(df.bs == bs)[x] and list(df.fs == fs)[x] for x in range(len(df))]
        else:
            filtered_indices = [list(df.bs == bs)[x] for x in range(len(df))]
        dfs.append(df.loc[filtered_indices].reset_index(drop=True))

    dfs, xlabels, ylabels = safety_checks(dfs, args)
        
    #Filter by numjobs or iodepth
    for i, label in enumerate(xlabels):
        if label == 'iodepth_parsed':
            filtered_indices = [list(dfs[i].numjobs_parsed == 1)[x] for x in range(len(dfs[i]))]
            dfs[i] = dfs[i].loc[filtered_indices].reset_index(drop=True)
        elif label == 'numjobs_parsed':
            filtered_indices = [list(dfs[i].iodepth_parsed == 1)[x] for x in range(len(dfs[i]))]
            dfs[i] = dfs[i].loc[filtered_indices].reset_index(drop=True)
    
    return dfs, xlabels, ylabels

def graph_setup(axes, labels, args):
    for x in range(3):
        axes[x].spines['top'].set_visible(False)
        axes[x].spines['right'].set_visible(False)
        axes[x].yaxis.set_major_formatter(FuncFormatter(my_format))
        axes[x].set_box_aspect(args.aspect_ratio)

    try:
        for x in range(3):
            axes[x].set_xlabel(std.get_feature(xlabels[x]))
            axes[x].set_ylabel(std.get_feature(ylabels[x]))
    except:
        for x in range(3):
            axes[x].set_xlabel(get_feature(xlabels[x]))
            axes[x].set_ylabel(get_feature(ylabels[x]))
        
    try:
        axes[0].set_xscale('log', basex=2)
        axes[1].set_xscale('log', basex=2)
        axes[2].set_xscale('log', basex=2)
    except:
        axes[0].set_xscale('log', base=2)
        axes[1].set_xscale('log', base=2)
        axes[2].set_xscale('log', base=2)
    
    return axes


###############################################################################################################

parser = argparse.ArgumentParser(prog='CSL Single Plot Generator',
                                 description=
                                 dedent('''
                                 Takes a <data_file> from the aggregate.py script output and graphs a triple plot based on preset or manual behavior.
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
                    default='watts_mean',
                    choices=['cpu_util', 'iops', 'watts_mean', 'iopj'],
                    help='Premade plots (2nd Plot)')

parser.add_argument('-p3', '--preset3',
                    type=str,
                    default='iopj',
                    choices=['cpu_util', 'iops', 'watts_mean', 'iopj'],
                    help='Premade plots (3rd Plot)')

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

parser.add_argument('-px3',
                    type=str,
                    default='numjobs',
                    choices=['numjobs', 'iodepth'],
                    help='What to plot on preset\'s x-axis (3rd Plot)')

parser.add_argument('-fs1',
                    type=str,
                    default='xfs',
                    nargs='?',
                    choices=['xfs', 'ext4', 'f2fs', 'all'],
                    help='Which filesystem specific data is being graphed (1st Plot)')

parser.add_argument('-fs2',
                    type=str,
                    default='xfs',
                    nargs='?',
                    choices=['xfs', 'ext4', 'f2fs', 'all'],
                    help='Which filesystem specific data is being graphed (2nd Plot)')

parser.add_argument('-fs3',
                    type=str,
                    default='xfs',
                    nargs='?',
                    choices=['xfs', 'ext4', 'f2fs', 'all'],
                    help='Which filesystem specific data is being graphed (3rd Plot)')

parser.add_argument('-afa1', '--all-fs-api1',
                    type=str,
                    default='io_uring int',
                    nargs='?',
                    choices=['io_uring int', 'io_uring cpool', 'io_uring spoll', 'io_uring both', 'pvsync2 int', 'pvsync2 cpoll', 'libaio int', 'spdk cpoll', 'posixaio int'],
                    help='Which API to graph if -fs1=\'all\'')

parser.add_argument('-afa2', '--all-fs-api2',
                    type=str,
                    default='io_uring int',
                    nargs='?',
                    choices=['io_uring int', 'io_uring cpool', 'io_uring spoll', 'io_uring both', 'pvsync2 int', 'pvsync2 cpoll', 'libaio int', 'spdk cpoll', 'posixaio int'],
                    help='Which API to graph if -fs2=\'all\'')

parser.add_argument('-afa3', '--all-fs-api3',
                    type=str,
                    default='io_uring int',
                    nargs='?',
                    choices=['io_uring int', 'io_uring cpool', 'io_uring spoll', 'io_uring both', 'pvsync2 int', 'pvsync2 cpoll', 'libaio int', 'spdk cpoll', 'posixaio int'],
                    help='Which API to graph if -fs3=\'all\'')


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

parser.add_argument('-x3',
                    type=str,
                    help='**Manual Plotting** x-axis (3rd Plot)')

parser.add_argument('-y3',
                    type=str,
                    help='**Manual Plotting** y-axis (3rd Plot)')

parser.add_argument('-g', '--grey', '--gray',
                    type=bool,
                    nargs='?',
                    default=False,
                    const=True,
                    help='Converts and plots as greyscale')

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

parser.add_argument('--figsize',
                    type=int,
                    nargs=2,
                    default=(9, 4.5),
                    help='Figure size (w, h)')

parser.add_argument('--bs1',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size (1st Plot)')

parser.add_argument('--bs2',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size (2nd Plot)')

parser.add_argument('--bs3',
                    type=str,
                    default='4K',
                    choices=['4K', '16K', '128K'],
                    help='Filter by block size (3rd Plot)')

parser.add_argument('-t1',
                    type=str,
                    help='Title (1st Plot)')

parser.add_argument('-t2',
                    type=str,
                    help='Title (2nd Plot)')

parser.add_argument('-t3',
                    type=str,
                    help='Title (3rd Plot)')

parser.add_argument('--legend-loc',
                    type=str,
                    default='upper center',
                    help='Sets the legend\s loc parameter ... default=\'upper center\'')

parser.add_argument('--subplots-wspace',
                    type=float,
                    default=0.4,
                    help='Sets the subplot\'s wspace ... default=0.4')

parser.add_argument('--legend-bbox-to-anchor',
                    type=tuple,
                    default=(0.5, -0.15),
                    help='Sets the legend\s bbox_to_anchor parameter ... default=(0.5, -0.15)')

parser.add_argument('--aspect-ratio',
                    type=str,
                    default='None',
                    help='Sets subplot\'s aspect ratio (ex: \'1/2\' or \'0.5\')')

### Parse CLI Arguments
args = parser.parse_args()
args.px1 += '_parsed'
args.px2 += '_parsed'
args.px3 += '_parsed'

try:
    if args.aspect_ratio == 'None':
        args.aspect_ratio = None
    else:
        args.aspect_ratio = float(eval(args.aspect_ratio))
except:
    raise Exception(f'Bad input to args.aspect-ratio: \"{args.aspect_ratio}\"')

### Read and parse the dataset
df = pd.read_csv(args.data_file)
df = parse_name_col(df)

###Special Functionality
if args.columns:
    for x in np.unique(df.columns.values):
        print(x)
    exit()
elif args.debug:
    breakpoint()

dfs, xlabels, ylabels = filter_and_safety(df, args)
    
fig, axes = plt.subplots(1, 3, figsize=tuple(args.figsize))
axes = graph_setup(axes, (xlabels, ylabels), args)

### Plotting
filesystems = (args.fs1, args.fs2, args.fs3)

for x in range(3):
    if filesystems[x] == 'all':
        fs_api_choice = (args.all_fs_api1, args.all_fs_api2, args.all_fs_api3)
        dfs[x] = dfs[x][dfs[x].ioengine == fs_api_choice[x]]
        ioengine = dfs[x].ioengine[0]
        
        for fs, group in dfs[x].groupby('fs'):
            f = group.sort_values(xlabels[x]).reset_index(drop=True)
            
            try:
                style = std.get_fs_style(fs)
                #if args.grey:
                #    style['color'] = std.convert_greyscale(*style.get('color'))
                
                axes[x].plot(f[xlabels[x]], f[ylabels[x]], label=fs, **style)

            except:
                style = get_fs_style(fs)
                #if args.grey:
                #    style['color'] = convert_greyscale(*style.get('color'))
                    
                axes[x].plot(f[xlabels[x]], f[ylabels[x]], label=fs, **style)

    else:
        for ioengine, group in dfs[x].groupby('ioengine'):
            f = group.sort_values(xlabels[x]).reset_index(drop=True)
            
            try:
                style = std.get_style(ioengine, gscale=args.grey)
                if args.grey:
                    style['color'] = std.convert_greyscale(*style.get('color'))
                
                axes[x].plot(f[xlabels[x]], f[ylabels[x]], marker=std.Markers.get(ioengine), markersize=4, label=std.get_label(ioengine), **style)

            except:
                style = get_style(ioengine, gscale=args.grey)
                if args.grey:
                    style['color'] = convert_greyscale(*style.get('color'))
                    
                axes[x].plot(f[xlabels[x]], f[ylabels[x]], marker=Markers.get(ioengine), markersize=4, label=get_label(ioengine), **style)
                
    if(ylabels[x] == 'watts_mean'):
        axes[x].set_ylim(40, dfs[x][ylabels[x]].max() * 1.05)
    else:
        axes[x].set_ylim(0, dfs[x][ylabels[x]].max() * 1.05)

### Adding final graph settings
x_ticks = [np.sort(dfs[0][xlabels[0]].unique()), np.sort(dfs[1][xlabels[1]].unique()), np.sort(dfs[2][xlabels[2]].unique())]

titles = [args.t1, args.t2, args.t3]
for x in range(3):
    axes[x].grid(axis='both', ls='--', alpha=0.2)
    axes[x].xaxis.set_major_formatter(ScalarFormatter())
    axes[x].set_xticks(x_ticks[x])

    if titles[x]:
        axes[x].set_title(titles[x])

axes[1].legend(loc=args.legend_loc, bbox_to_anchor=args.legend_bbox_to_anchor, labelspacing=0.15, handletextpad=0.15, frameon=False, fancybox=False, shadow=False, ncol=ceil(len(np.unique(df['ioengine'])) / 2))

plt.tight_layout()
plt.subplots_adjust(wspace=args.subplots_wspace)

if args.output:
    plt.savefig(args.output, bbox_inches='tight', transparent=True)
else:
    print('\n\n***LEGEND MIGHT NOT SHOW CORRECTLY UNLESS SAVED***\n\n')
    plt.show()
