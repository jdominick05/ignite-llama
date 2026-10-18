#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

//
// common_ngram_suffix
//
// per-sequence self-speculative drafter: indexes the start position of every k-gram of the current request
// (prompt + accepted tokens), most recent occurrence wins. To draft, the last k tokens (ending with id_last) are
// looked up, the earlier occurrence is verified token by token (hash collisions never produce a draft) and the
// tokens that followed it are proposed.
//
// policies:
//   match - extend the match backwards while the tokens agree (match length m, k <= m <= l_max) and draft
//           max(l_min, m) tokens
//   fixed - always draft l_max tokens
//   adapt - start at l_min, double after a fully accepted draft, otherwise accepted + 1
// the draft is always limited to the tokens that exist after the earlier occurrence.
//
// no llama dependencies: tokens are plain int32_t
//

enum common_ngram_suffix_policy {
    COMMON_NGRAM_SUFFIX_POLICY_MATCH,
    COMMON_NGRAM_SUFFIX_POLICY_FIXED,
    COMMON_NGRAM_SUFFIX_POLICY_ADAPT,
};

// "match", "fixed", "adapt"; returns false if unknown
bool        common_ngram_suffix_policy_from_str(const char * str, common_ngram_suffix_policy & policy);
const char * common_ngram_suffix_policy_to_str(common_ngram_suffix_policy policy);

struct common_ngram_suffix {
    using token_t = int32_t;

    common_ngram_suffix(uint16_t k, uint16_t l_min, uint16_t l_max, common_ngram_suffix_policy policy);

    // index a new prompt (drops everything indexed before)
    void begin(const token_t * prompt, size_t n_prompt);

    // ctx = prompt[0, n_prompt) + id_last; prompt holds every token before id_last (a prefix-extension of the
    // tokens seen so far). result is cleared and receives the draft (possibly empty)
    void draft(const token_t * prompt, size_t n_prompt, token_t id_last, std::vector<token_t> & result);

    // n_accepted of the n_drafted tokens of the last draft were accepted (only used by the adapt policy)
    void accept(size_t n_accepted, size_t n_drafted);

    // 64-bit polynomial hash of k tokens: res = res*6364136223846793005 + tok (mod 2^64)
    static uint64_t hash(const token_t * tokens, size_t k);

    size_t n_entries() const { return used; }

private:
    void     clear();
    void     insert(uint64_t key, uint32_t pos);
    uint32_t find(uint64_t key) const; // NONE if not found
    void     grow();
    void     index_range(const token_t * tokens, size_t lo, size_t hi); // window starts [lo, hi)

    static constexpr uint32_t NONE = UINT32_MAX;

    uint16_t k;
    uint16_t l_min;
    uint16_t l_max;
    common_ngram_suffix_policy policy;

    // open addressing hash table: k-gram hash -> latest start position
    std::vector<uint64_t> keys;
    std::vector<uint32_t> vals; // NONE = empty slot
    size_t used = 0;

    size_t i_last   = 0;    // next window start to index
    size_t next_len = 0;    // adapt policy: length of the next draft

    // consistency check for callers that change the context without begin(): last indexed token
    size_t  chk_pos = SIZE_MAX;
    token_t chk_tok = 0;
};
