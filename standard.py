#!/usr/bin/env python3

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
        if gscale:
            return specific_style_greyscale[name]
        else:
            return specific_style[name]

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
