import subprocess, os, time

opts = [{'compression': c, 'packing': p} for c in ['snappy'] for p in ['none', 'half', 'full']]

def extract_write_pack_percents(prof):
    with open(prof) as f:
        lines = [line.split() for line in f]
    total_traces = sum(int(l[1]) for l in lines)
    write_traces = [int(l[1]) for l in lines if l[0].endswith('write') or l[0].endswith('compress')]
    pack_traces = [int(l[1]) for l in lines if l[0].endswith('pack')]
    return sum(write_traces) / total_traces, sum(pack_traces) / total_traces



def run_options(compression, packing):
    filename = '{}_{}'.format(compression, packing)
    tick = time.time()
    subprocess.call("python -m flamegraph -o {}.prof --filter=preprocess main.py preprocess data/kgs-mini --compression={} --packing={}".format(filename, compression, packing), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tock = time.time()
    subprocess.call("../Flamegraph/flamegraph.pl {}.prof > {}.svg".format(filename, filename), shell=True)
    compressed_filesize = os.path.getsize('processed_data/test.chunk.gz') + os.path.getsize('processed_data/train0.chunk.gz')
    total_time = tock - tick
    write_frac, pack_frac = extract_write_pack_percents("{}.prof".format(filename))
    print("%s\t%s\t%.2f\t%.2f\t%d" % (compression, packing, write_frac * total_time, pack_frac * total_time, compressed_filesize))

for opt_set in opts:
    run_options(**opt_set)

