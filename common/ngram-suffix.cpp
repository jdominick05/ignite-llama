#include "ngram-suffix.h"

#include <algorithm>
#include <cstring>

//
// common_ngram_suffix
//

bool common_ngram_suffix_policy_from_str(const char * str, common_ngram_suffix_policy & policy) {
    if (std::strcmp(str, "match") == 0) { policy = COMMON_NGRAM_SUFFIX_POLICY_MATCH; return true; }
    if (std::strcmp(str, "fixed") == 0) { policy = COMMON_NGRAM_SUFFIX_POLICY_FIXED; return true; }
    if (std::strcmp(str, "adapt") == 0) { policy = COMMON_NGRAM_SUFFIX_POLICY_ADAPT; return true; }
    return false;
}

const char * common_ngram_suffix_policy_to_str(common_ngram_suffix_policy policy) {
    switch (policy) {
        case COMMON_NGRAM_SUFFIX_POLICY_MATCH: return "match";
        case COMMON_NGRAM_SUFFIX_POLICY_FIXED: return "fixed";
        case COMMON_NGRAM_SUFFIX_POLICY_ADAPT: return "adapt";
    }
    return "unknown";
}

common_ngram_suffix::common_ngram_suffix(uint16_t k, uint16_t l_min, uint16_t l_max, common_ngram_suffix_policy policy)
    : k(std::max<uint16_t>(1, k)), l_min(l_min), l_max(l_max), policy(policy), next_len(l_min) {
    keys.resize(1u << 12);
    vals.assign(keys.size(), NONE);
}

uint64_t common_ngram_suffix::hash(const token_t * tokens, size_t k) {
    uint64_t res = 0;
    for (size_t i = 0; i < k; ++i) {
        res = res*6364136223846793005ULL + (uint64_t) (int64_t) tokens[i];
    }
    return res;
}

static inline size_t ngram_suffix_slot(uint64_t key, size_t mask) {
    // the polynomial hash is weak in its low bits for short k-grams: mix before masking
    return (size_t) ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

void common_ngram_suffix::clear() {
    std::fill(vals.begin(), vals.end(), NONE);
    used = 0;
}

void common_ngram_suffix::grow() {
    std::vector<uint64_t> keys_old;
    std::vector<uint32_t> vals_old;
    keys_old.swap(keys);
    vals_old.swap(vals);

    keys.resize(keys_old.size()*2);
    vals.assign(keys.size(), NONE);
    used = 0;

    for (size_t i = 0; i < keys_old.size(); ++i) {
        if (vals_old[i] != NONE) {
            insert(keys_old[i], vals_old[i]);
        }
    }
}

void common_ngram_suffix::insert(uint64_t key, uint32_t pos) {
    if ((used + 1)*2 > keys.size()) {
        grow();
    }

    const size_t mask = keys.size() - 1;
    for (size_t i = ngram_suffix_slot(key, mask); ; i = (i + 1) & mask) {
        if (vals[i] == NONE) {
            keys[i] = key;
            vals[i] = pos;
            used++;
            return;
        }
        if (keys[i] == key) {
            vals[i] = pos; // most recent occurrence wins
            return;
        }
    }
}

uint32_t common_ngram_suffix::find(uint64_t key) const {
    const size_t mask = keys.size() - 1;
    for (size_t i = ngram_suffix_slot(key, mask); ; i = (i + 1) & mask) {
        if (vals[i] == NONE) {
            return NONE;
        }
        if (keys[i] == key) {
            return vals[i];
        }
    }
}

void common_ngram_suffix::index_range(const token_t * tokens, size_t lo, size_t hi) {
    for (size_t s = lo; s < hi; ++s) {
        insert(hash(tokens + s, k), (uint32_t) s);
    }
}

void common_ngram_suffix::begin(const token_t * prompt, size_t n_prompt) {
    clear();

    // reserve for the prompt at once instead of growing step by step
    while (n_prompt*2 > keys.size()) {
        keys.resize(keys.size()*2);
        vals.assign(keys.size(), NONE);
    }

    i_last   = n_prompt >= k ? n_prompt - k + 1 : 0;
    next_len = l_min;
    chk_pos  = SIZE_MAX;

    if (i_last > 0) {
        index_range(prompt, 0, i_last);

        chk_pos = n_prompt - 1;
        chk_tok = prompt[chk_pos];
    }
}

void common_ngram_suffix::draft(const token_t * prompt, size_t n_prompt, token_t id_last, std::vector<token_t> & result) {
    result.clear();

    const size_t end = n_prompt + 1; // tokens known so far, including id_last
    if (end < (size_t) k + 1) {
        return;
    }

    // the context was changed without begin() (shrunk or rewritten): index it from scratch
    if (i_last > end - k || (chk_pos != SIZE_MAX && (chk_pos >= n_prompt || prompt[chk_pos] != chk_tok))) {
        clear();
        i_last  = 0;
        chk_pos = SIZE_MAX;
    }

    // index the windows that start before the lookup window (they lie entirely within prompt)
    if (i_last < end - k) {
        index_range(prompt, i_last, end - k);
        i_last  = end - k;
        chk_pos = i_last + k - 2;
        chk_tok = prompt[chk_pos];
    }

    auto at = [&](size_t i) -> token_t {
        return i < n_prompt ? prompt[i] : id_last;
    };

    // lookup window: the last k tokens, ending with id_last
    const size_t w0 = end - k;

    uint64_t key = 0;
    for (size_t i = 0; i < k; ++i) {
        key = key*6364136223846793005ULL + (uint64_t) (int64_t) at(w0 + i);
    }

    const uint32_t p32 = find(key);
    if (p32 == NONE || (size_t) p32 >= w0) {
        return;
    }
    const size_t p = p32;

    // verify (no hash collisions accepted): the latest occurrence under this hash must match, no fallback
    for (size_t i = 0; i < k; ++i) {
        if (at(p + i) != at(w0 + i)) {
            return;
        }
    }

    const size_t src = p + k; // continuation of the earlier occurrence (src < end)

    size_t L = 0;
    switch (policy) {
        case COMMON_NGRAM_SUFFIX_POLICY_FIXED:
            L = l_max;
            break;
        case COMMON_NGRAM_SUFFIX_POLICY_MATCH: {
            size_t m = k;
            while (m < l_max && p >= (m - k) + 1 && at(p - (m - k) - 1) == at(end - m - 1)) {
                m++;
            }
            L = std::max<size_t>(l_min, std::min<size_t>(l_max, m));
        } break;
        case COMMON_NGRAM_SUFFIX_POLICY_ADAPT:
            L = next_len;
            break;
    }

    L = std::min(L, end - src);

    result.resize(L);
    for (size_t i = 0; i < L; ++i) {
        result[i] = at(src + i);
    }
}

void common_ngram_suffix::accept(size_t n_accepted, size_t n_drafted) {
    if (policy == COMMON_NGRAM_SUFFIX_POLICY_ADAPT && n_drafted > 0) {
        next_len = n_accepted == n_drafted
            ? std::min<size_t>(l_max, n_drafted*2)
            : std::max<size_t>(l_min, n_accepted + 1);
    }
}
