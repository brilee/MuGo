import os
from load_data_sets import process_raw_data, timer

opts = [{'compression': c, 'packing': p} for p in ['full', 'none', 'half'] for c in ['none', 'gzip9', 'gzip6', 'snappy']]


def run_options(compression, packing):
    print("%s\t%s\t" % (packing, compression), end='|')
    with timer():
        process_raw_data('data/kgs-mini', compression=compression, packing=packing)
    compressed_filesize = os.path.getsize('processed_data/all_data')
    print(compressed_filesize, end='|')
    print()

processed_dir = 'processed_data'

processed_dir = os.path.join(os.getcwd(), processed_dir)
if not os.path.isdir(processed_dir):
    os.mkdir(processed_dir)

print("| bitpack | compression | time to process (s) | time to convert to bytes (s) | time to compress, write (s) | total time (s) | output size (bytes) |")
for opt_set in opts:
    run_options(**opt_set)

