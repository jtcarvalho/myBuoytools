#!/bin/sh

bsub -q s_long -P R000 -o log.out -e log.err -M 5GB -R rusage[mem=5GB]  python ww3EMODNET_scatter.py

