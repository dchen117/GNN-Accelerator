// Ramulator 2.1 replay driver for megasim (S-13).
//
// Replays the DRAM request trace of one timing-model run through Ramulator's HBM model and
// reports, for every transfer ("batch") of the timing model, when its first and its last
// request completed. megasim/dram/ramulator.py writes the inputs, runs this, and reads the
// outputs; the timing model then uses the measured latencies (fixed-point iteration, S-13).
//
//   replay <config.yaml> <trace prefix> <out prefix>
//
// Inputs (raw little-endian arrays, written by ramulator.py):
//   <trace>.batch_t   int64  [B]  issue cycle of each batch (accelerator clock = DRAM clock)
//   <trace>.batch_ptr int64  [B+1] requests of batch b are [ptr[b], ptr[b+1]), sorted by t
//   <trace>.addr      uint64 [R]  byte address of each 64 B line
//   <trace>.write     uint8  [R]  1 = write
// Outputs:
//   <out>.first int64 [B]  cycle the first request of the batch completed (-1: empty batch)
//   <out>.last  int64 [B]  cycle the last request completed
//   <out>.stats        Ramulator's statistics (YAML), plus driver counters at the end
//
// Model of the requester (ours, S-13): the accelerator's DMA keeps one FIFO per HBM channel.
// A batch's lines enter the FIFOs at its issue cycle, in trace order. Every cycle, each
// channel FIFO offers its head line to Ramulator, up to kTxPerChannelCycle 32 B transactions
// per channel (a 64 B line is two HBM1 transactions: BL2 x 128 bit = 32 B); a refused
// transaction (controller queue full) is retried the next cycle. Reads complete when
// Ramulator's callback fires (read data returned); writes complete when the WR command
// issues (Ramulator's convention; the data is in the device's write path then).

#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

constexpr int kLineBytes = 64;
constexpr int kTxBytes = 32;              // HBM1: 128 bit x BL2
constexpr int kTxPerLine = kLineBytes / kTxBytes;
constexpr int kTxPerChannelCycle = 2;     // DMA issue width per channel (one line per cycle)
constexpr int kChannels = 8;              // must match hbm_config.py
constexpr int kChannelShift = 6;          // CacheLineInterleave with interleave_bits = 1
// HBM1_4Gb per channel: 4 bank groups x 4 banks x 16384 rows x 64 transactions (2 KB rows)
constexpr int kBgBits = 2, kBankBits = 2, kRowBits = 14, kColBits = 6;

// Address mapping (ours, S-13). From the physical byte address A, MSB to LSB:
//     row(14) | bank(2) | column[5:1](5) | bankgroup(2) | channel(3) | column[0](1) | byte(5)
// i.e. 64 B lines go round the 8 channels; a channel's consecutive lines go round its 4 bank
// groups (consecutive column commands to different bank groups need only tCCD_S = 1, to the
// same group tCCD_L = 2), then along the open 2 KB row; both 32 B halves of a line are in the
// same row. Ramulator's controller uses RoBaRaCoCh (column bits lowest after the channel), so
// we re-encode A into the address whose RoBaRaCoCh decoding gives exactly these fields.
uint64_t remap(uint64_t A) {
  uint64_t half = (A >> 5) & 1;
  uint64_t ch = (A >> kChannelShift) & (kChannels - 1);
  uint64_t rest = A >> 9;                               // line index inside the channel
  uint64_t bg = rest & ((1 << kBgBits) - 1);
  rest >>= kBgBits;
  uint64_t colhi = rest & ((1 << (kColBits - 1)) - 1);
  rest >>= kColBits - 1;
  uint64_t bank = rest & ((1 << kBankBits) - 1);
  rest >>= kBankBits;
  uint64_t row = rest & ((1 << kRowBits) - 1);
  uint64_t col = (colhi << 1) | half;
  // RoBaRaCoCh after the channel is stripped: tx index = col | bg | bank | row (LSB first)
  uint64_t tx = col | (bg << kColBits) | (bank << (kColBits + kBgBits)) |
                (row << (kColBits + kBgBits + kBankBits));
  uint64_t intra = tx << 5;                             // + byte offset 0
  // CacheLineInterleave(interleave_bits = 1): channel bits sit at [8:6]
  return ((intra >> 6) << 9) | (ch << 6) | (intra & 63);
}

template <typename T>
std::vector<T> read_array(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<T> v(n / sizeof(T));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

template <typename T>
void write_array(const std::string& path, const std::vector<T>& v) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: replay <config.yaml> <trace prefix> <out prefix>\n";
    return 2;
  }
  const std::string cfg_path = argv[1], tr = argv[2], out = argv[3];

  auto batch_t = read_array<int64_t>(tr + ".batch_t");
  auto batch_ptr = read_array<int64_t>(tr + ".batch_ptr");
  auto addr = read_array<uint64_t>(tr + ".addr");
  auto wr = read_array<uint8_t>(tr + ".write");
  const int64_t B = static_cast<int64_t>(batch_t.size());
  if (static_cast<int64_t>(batch_ptr.size()) != B + 1 || addr.size() != wr.size() ||
      batch_ptr[B] != static_cast<int64_t>(addr.size()))
    throw std::runtime_error("inconsistent trace arrays");

  auto config = Ramulator::Config::parse_config_file(cfg_path);
  auto* frontend = Ramulator::Factory::create_frontend(config);
  auto* mem = Ramulator::Factory::create_memory_system(config);
  frontend->connect_memory_system(mem);
  mem->connect_frontend(frontend);
  if (mem->get_tx_bytes() != kTxBytes) throw std::runtime_error("unexpected transaction size");

  std::vector<int64_t> first(B, -1), last(B, -1);
  std::vector<int64_t> outstanding(B, 0);       // transactions not yet completed per batch
  std::vector<int64_t> req_batch(addr.size());
  for (int64_t b = 0; b < B; b++)
    for (int64_t r = batch_ptr[b]; r < batch_ptr[b + 1]; r++) req_batch[r] = b;

  int64_t clk = 0;
  auto done = [&](int64_t b) {
    if (first[b] < 0) first[b] = clk;
    last[b] = clk;
    outstanding[b]--;
  };

  // Batches enter in issue order (ties: trace order; ramulator.py sorts stably).
  std::vector<int64_t> order(B);
  for (int64_t b = 0; b < B; b++) order[b] = b;
  for (int64_t b = 1; b < B; b++)
    if (batch_t[b] < batch_t[b - 1]) throw std::runtime_error("batches must be sorted by issue time");

  std::deque<int64_t> fifo[kChannels];          // request indices
  int half[kChannels] = {0};                     // transactions of the head line already sent
  int64_t next_batch = 0, in_fifo = 0, in_flight = 0, refused = 0;

  while (next_batch < B || in_fifo > 0 || in_flight > 0) {
    // 1. batches issued at this cycle enter the channel FIFOs
    while (next_batch < B && batch_t[next_batch] <= clk) {
      int64_t b = next_batch++;
      for (int64_t r = batch_ptr[b]; r < batch_ptr[b + 1]; r++) {
        fifo[(addr[r] >> kChannelShift) % kChannels].push_back(r);
        outstanding[b] += kTxPerLine;
        in_fifo++;
      }
    }
    // 2. each channel offers up to kTxPerChannelCycle transactions
    for (int c = 0; c < kChannels; c++) {
      for (int k = 0; k < kTxPerChannelCycle && !fifo[c].empty(); k++) {
        int64_t r = fifo[c].front();
        int64_t b = req_batch[r];
        Ramulator::Addr_t a = static_cast<Ramulator::Addr_t>(remap(addr[r] + half[c] * kTxBytes));
        int type = wr[r] ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
        bool ok = frontend->receive_external_requests(
            type, a, 0,
            [&, b](Ramulator::Request&) {
              done(b);
              in_flight--;
            },
            kTxBytes);
        if (!ok) {
          refused++;
          break;
        }
        in_flight++;
        if (++half[c] == kTxPerLine) {
          half[c] = 0;
          fifo[c].pop_front();
          in_fifo--;
        }
      }
    }
    // 3. one DRAM command clock (= one accelerator cycle, both 1 GHz)
    mem->tick();
    clk++;
    // jump over idle stretches only when nothing at all is pending (keeps refresh honest:
    // the controller is still ticked through them, but cheaply in a tight loop)
    if (in_fifo == 0 && in_flight == 0 && next_batch < B && batch_t[next_batch] > clk) {
      while (clk < batch_t[next_batch]) {
        mem->tick();
        clk++;
      }
    }
  }

  for (int64_t b = 0; b < B; b++)
    if (outstanding[b] != 0) throw std::runtime_error("batch did not complete");

  frontend->finalize();
  mem->finalize();
  write_array(out + ".first", first);
  write_array(out + ".last", last);
  std::ofstream st(out + ".stats");
  mem->print_stats(st);
  st << "driver:\n  cycles: " << clk << "\n  refused_tx: " << refused
     << "\n  lines: " << addr.size() << "\n  batches: " << B << "\n";
  return 0;
}
