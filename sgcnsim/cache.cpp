// Set-associative LRU cache over a sequence of 64 B line addresses (SGCN global cache, G-05).
// Built into sgcnsim/_cache.so by sgcnsim/build.sh and called from sgcnsim/cache.py via ctypes.
//   lru(lines, n, sets, ways, hit): hit[i] = 1 if access i hits, else 0 (the line is filled).
// Set index = line % sets; LRU by a per-access timestamp. Reads only (writes bypass, G-05).
#include <cstdint>
#include <vector>

extern "C" void lru(const uint64_t* lines, int64_t n, int64_t sets, int64_t ways, uint8_t* hit) {
    std::vector<uint64_t> tag(sets * ways, ~0ULL);
    std::vector<int64_t> stamp(sets * ways, -1);
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t l = lines[i];
        const int64_t base = (int64_t)(l % (uint64_t)sets) * ways;
        int64_t victim = base, found = -1;
        for (int64_t w = base; w < base + ways; ++w) {
            if (tag[w] == l) { found = w; break; }
            if (stamp[w] < stamp[victim]) victim = w;      // empty ways have stamp -1
        }
        if (found >= 0) { hit[i] = 1; stamp[found] = i; }
        else { hit[i] = 0; tag[victim] = l; stamp[victim] = i; }
    }
}
