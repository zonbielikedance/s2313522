Traceback (most recent call last):
  File "/usr/local/bin/pssh", line 106, in <module>
    opts, args = parse_args()
  File "/usr/local/bin/pssh", line 49, in parse_args
    parser = option_parser()
  File "/usr/local/bin/pssh", line 31, in option_parser
    parser = common_parser()
  File "/usr/local/lib/python3.9/site-packages/psshlib/cli.py", line 22, in common_parser
    version=version.VERSION)
AttributeError: module 'version' has no attribute 'VERSION'

Authorized users only. All activities may be monitored and reported.

Authorized users only. All activities may be monitored and reported.
Traceback (most recent call last):
  File "/usr/local/bin/pscp", line 92, in <module>
    opts, args = parse_args()
  File "/usr/local/bin/pscp", line 39, in parse_args
    parser = option_parser()
  File "/usr/local/bin/pscp", line 28, in option_parser
    parser = common_parser()
  File "/usr/local/lib/python3.9/site-packages/psshlib/cli.py", line 22, in common_parser
    version=version.VERSION)
AttributeError: module 'version' has no attribute 'VERSION'
load data /anndata/DEEP100K.query.fbin
dimension: 96  number:10000  size_per_element:4
load data /anndata/DEEP100K.gt.query.100k.top100.bin
dimension: 100  number:10000  size_per_element:4
load data /anndata/DEEP100K.base.100k.fbin
dimension: 96  number:100000  size_per_element:4
