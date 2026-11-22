"""Ramulator 2.1 configuration of MEGA's DRAM (Q-29, S-13).

HBM1 (JESD235), 1 stack, 8 channels × 128 bit at 2 Gb/s/pin = 8 × 32 GB/s = 256 GB/s, the
bandwidth Tab.IV / §VI-A-3 states. The command clock is 1 GHz (tCK = 1000 ps), the same as the
accelerator clock, so the replay driver ticks the memory once per accelerator cycle.

    FR-FCFS scheduler, open-row policy, all-bank refresh.
    Address mapping (S-13, done in replay.cpp's remap(), Ramulator decodes it with RoBaRaCoCh):
        row | bank | column[5:1] | bankgroup | channel | column[0] | byte
    64 B lines (one MEGA line = two 32 B HBM1 bursts, BL2 × 128 bit) go round the 8 channels,
    then round a channel's 4 bank groups, then along the open 2 KB row.

Export (done by megasim/dram/build.sh):
    PYTHONPATH=$RAMULATOR/python python -m ramulator export megasim/dram/hbm_config.py \
        -o megasim/dram/hbm1_256GBps.yaml
"""

import ramulator

CHANNELS = 8

frontend = ramulator.frontend.External(clock_ratio=1)
dram = ramulator.dram.HBM1(org_preset="HBM1_4Gb", timing_preset="HBM1_2Gbps")
ctrl = ramulator.controller.HBM12(
    dram=dram,
    scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),   # see replay.cpp remap()
)
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[ctrl] * CHANNELS,
    # interleave_bits=1: 2 × 32 B transactions = one 64 B line per channel
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=1),
)
sim = ramulator.Simulation(frontend, mem)
