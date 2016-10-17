import subprocess, os, time

opts = [{'compression': c, 'packing': p} for p in ['none', 'half', 'full'] for c in ['none', 'gzip9', 'gzip6', 'snappy']]

def extract_write_pack_percents(prof):
    with open(prof) as f:
        lines = [line.split() for line in f]
    total_traces = sum(int(l[1]) for l in lines)
    write_traces = [int(l[1]) for l in lines if l[0].endswith('write') or l[0].endswith('compress')]
    pack_traces = [int(l[1]) for l in lines if l[0].endswith('pack')]
    pack = sum(pack_traces)
    write = sum(write_traces)
    remainder = total_traces - pack - write
    return remainder / total_traces, pack / total_traces, write / total_traces


def run_options(compression, packing):
    filename = '{}_{}'.format(compression, packing)
    tick = time.time()
    subprocess.call("python -m flamegraph -o {}.prof --filter=preprocess main.py preprocess data/kgs-mini --compression={} --packing={}".format(filename, compression, packing), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tock = time.time()
    subprocess.call("../Flamegraph/flamegraph.pl {}.prof > {}.svg".format(filename, filename), shell=True)
    compressed_filesize = os.path.getsize('processed_data/test.chunk.gz') + os.path.getsize('processed_data/train0.chunk.gz')
    total_time = tock - tick
    remainder_frac, write_frac, pack_frac = extract_write_pack_percents("{}.prof".format(filename))
    print("%s\t%s\t%.2f\t%.2f\t%.2f\t%d" % (packing, compression, remainder_frac * total_time, write_frac * total_time, pack_frac * total_time, compressed_filesize))

for opt_set in opts:
    print("packing | compression | processing | bitpack | compress+write | output size")
    run_options(**opt_set)

