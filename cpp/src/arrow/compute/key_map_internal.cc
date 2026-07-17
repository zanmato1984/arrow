// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/compute/key_map_internal.h"

#include <algorithm>
#include <bit>
#include <cstdint>

#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/ubsan.h"

namespace arrow {

using internal::CpuInfo;

namespace compute {

constexpr uint64_t kHighBitOfEachByte = 0x8080808080808080ULL;

// Scan bytes in block in reverse and stop as soon
// as a position of interest is found.
//
// Positions of interest:
// a) slot with a matching stamp is encountered,
// b) first empty slot is encountered,
// c) we reach the end of the block.
//
// Optionally an index of the first slot to start the search from can be specified. In
// this case slots before it will be ignored.
//
template <bool use_start_slot>
inline void SwissTable::search_block(uint64_t block, int stamp, int start_slot,
                                     int* out_slot, int* out_match_found) const {
  // Filled slot bytes have the highest bit set to 0 and empty slots are equal to 0x80.
  uint64_t block_high_bits = block & kHighBitOfEachByte;

  // Replicate 7-bit stamp to all non-empty slots, leaving zeroes for empty slots.
  uint64_t stamp_pattern = stamp * ((block_high_bits ^ kHighBitOfEachByte) >> 7);

  // If we xor this pattern with block status bytes we get in individual bytes:
  // a) 0x00, for filled slots matching the stamp,
  // b) 0x00 < x < 0x80, for filled slots not matching the stamp,
  // c) 0x80, for empty slots.
  uint64_t block_xor_pattern = block ^ stamp_pattern;

  // If we then add 0x7f to every byte, we get:
  // a) 0x7F
  // b) 0x80 <= x < 0xFF
  // c) 0xFF
  uint64_t match_base = block_xor_pattern + ~kHighBitOfEachByte;

  // The highest bit now tells us if we have a match (0) or not (1).
  // We will negate the bits so that match is represented by a set bit.
  uint64_t matches = ~match_base;

  // Clear 7 non-relevant bits in each byte.
  // Also clear bytes that correspond to slots that we were supposed to
  // skip due to provided start slot index.
  // Note: the highest byte corresponds to the first slot.
  if (use_start_slot) {
    matches &= kHighBitOfEachByte >> (8 * start_slot);
  } else {
    matches &= kHighBitOfEachByte;
  }

  // In case when there are no matches in slots and the block is full (no empty slots),
  // pretend that there is a match in the last slot.
  //
  matches |= (~block_high_bits & 0x80);

  // We get 0 if there are no matches
  *out_match_found = (matches == 0 ? 0 : 1);

  // Now if we or with the highest bits of the block and scan zero bits in reverse, we get
  // 8x slot index that we were looking for. This formula works in all three cases a), b)
  // and c).
  *out_slot = static_cast<int>(std::countl_zero(matches | block_high_bits) >> 3);
}

template <typename T, bool use_selection>
void SwissTable::extract_group_ids_imp(const int num_keys, const uint16_t* selection,
                                       const uint32_t* hashes, const uint8_t* local_slots,
                                       uint32_t* out_group_ids) const {
  if (log_blocks_ == 0) {
    DCHECK_EQ(sizeof(T), sizeof(uint8_t));
    for (int i = 0; i < num_keys; ++i) {
      uint32_t id = use_selection ? selection[i] : i;
      uint32_t group_id =
          block_data(/*block_id=*/0,
                     /*num_block_bytes=*/0)[bytes_status_in_block_ + local_slots[id]];
      out_group_ids[id] = group_id;
    }
  } else {
    int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
    DCHECK_EQ(sizeof(T) * 8, num_groupid_bits);
    int num_block_bytes = num_block_bytes_from_num_groupid_bits(num_groupid_bits);

    for (int i = 0; i < num_keys; ++i) {
      uint32_t id = use_selection ? selection[i] : i;
      uint32_t hash = hashes[id];
      uint32_t block_id = block_id_from_hash(hash, log_blocks_);
      const T* slots_base = reinterpret_cast<const T*>(
          block_data(block_id, num_block_bytes) + bytes_status_in_block_);
      uint32_t group_id = static_cast<uint32_t>(slots_base[local_slots[id]]);
      out_group_ids[id] = group_id;
    }
  }
}

void SwissTable::extract_group_ids(const int num_keys, const uint16_t* optional_selection,
                                   const uint32_t* hashes, const uint8_t* local_slots,
                                   uint32_t* out_group_ids) const {
  int num_processed = 0;
  // Optimistically use simplified lookup involving only a start block to find
  // a single group id candidate for every input.
#if defined(ARROW_HAVE_RUNTIME_AVX2) && defined(ARROW_HAVE_RUNTIME_BMI2)
  if ((hardware_flags_ & CpuInfo::AVX2) && CpuInfo::GetInstance()->HasEfficientBmi2() &&
      !optional_selection) {
    num_processed = extract_group_ids_avx2(num_keys, hashes, local_slots, out_group_ids);
  }
#endif
  int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
  switch (num_groupid_bits) {
    case 8:
      if (optional_selection) {
        extract_group_ids_imp<uint8_t, true>(num_keys, optional_selection, hashes,
                                             local_slots, out_group_ids);
      } else {
        extract_group_ids_imp<uint8_t, false>(
            num_keys - num_processed, nullptr, hashes + num_processed,
            local_slots + num_processed, out_group_ids + num_processed);
      }
      break;
    case 16:
      if (optional_selection) {
        extract_group_ids_imp<uint16_t, true>(num_keys, optional_selection, hashes,
                                              local_slots, out_group_ids);
      } else {
        extract_group_ids_imp<uint16_t, false>(
            num_keys - num_processed, nullptr, hashes + num_processed,
            local_slots + num_processed, out_group_ids + num_processed);
      }
      break;
    case 32:
      if (optional_selection) {
        extract_group_ids_imp<uint32_t, true>(num_keys, optional_selection, hashes,
                                              local_slots, out_group_ids);
      } else {
        extract_group_ids_imp<uint32_t, false>(
            num_keys - num_processed, nullptr, hashes + num_processed,
            local_slots + num_processed, out_group_ids + num_processed);
      }
      break;
    default:
      DCHECK(false);
  }
}

void SwissTable::init_slot_ids(const int num_keys, const uint16_t* selection,
                               const uint32_t* hashes, const uint8_t* local_slots,
                               const uint8_t* match_bitvector,
                               uint32_t* out_slot_ids) const {
  ARROW_DCHECK(selection);
  if (log_blocks_ == 0) {
    for (int i = 0; i < num_keys; ++i) {
      uint16_t id = selection[i];
      uint32_t match = ::arrow::bit_util::GetBit(match_bitvector, id) ? 1 : 0;
      uint32_t slot_id = local_slots[id] + match;
      out_slot_ids[id] = slot_id;
    }
  } else {
    for (int i = 0; i < num_keys; ++i) {
      uint16_t id = selection[i];
      uint32_t hash = hashes[id];
      uint32_t iblock = block_id_from_hash(hash, log_blocks_);
      uint32_t match = ::arrow::bit_util::GetBit(match_bitvector, id) ? 1 : 0;
      uint32_t slot_id = global_slot_id(iblock, local_slots[id] + match);
      out_slot_ids[id] = slot_id;
    }
  }
}

void SwissTable::init_slot_ids_for_new_keys(uint32_t num_ids, const uint16_t* ids,
                                            const uint32_t* hashes,
                                            uint32_t* slot_ids) const {
  int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
  int num_block_bytes = num_block_bytes_from_num_groupid_bits(num_groupid_bits);
  if (log_blocks_ == 0) {
    uint64_t block = *reinterpret_cast<const uint64_t*>(blocks_->mutable_data());
    uint32_t empty_slot =
        static_cast<uint32_t>(kSlotsPerBlock - std::popcount(block & kHighBitOfEachByte));
    for (uint32_t i = 0; i < num_ids; ++i) {
      int id = ids[i];
      slot_ids[id] = empty_slot;
    }
  } else {
    for (uint32_t i = 0; i < num_ids; ++i) {
      int id = ids[i];
      uint32_t hash = hashes[id];
      uint32_t iblock = block_id_from_hash(hash, log_blocks_);
      uint64_t block;
      for (;;) {
        block = *reinterpret_cast<const uint64_t*>(block_data(iblock, num_block_bytes));
        block &= kHighBitOfEachByte;
        if (block) {
          break;
        }
        iblock = (iblock + 1) & ((1 << log_blocks_) - 1);
      }
      uint32_t empty_slot = static_cast<int>(kSlotsPerBlock - std::popcount(block));
      slot_ids[id] = global_slot_id(iblock, empty_slot);
    }
  }
}

// Quickly filter out keys that have no matches based only on hash value and the
// corresponding starting 64-bit block of slot status bytes. May return false positives.
//
void SwissTable::early_filter_imp(const int num_keys, const uint32_t* hashes,
                                  uint8_t* out_match_bitvector,
                                  uint8_t* out_local_slots) const {
  // Clear the output bit vector
  memset(out_match_bitvector, 0, (num_keys + 7) / 8);

  // Based on the size of the table, prepare bit number constants.
  uint32_t stamp_mask = (1 << bits_stamp_) - 1;
  int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
  int num_block_bytes = num_block_bytes_from_num_groupid_bits(num_groupid_bits);

  for (int i = 0; i < num_keys; ++i) {
    // Extract from hash: block index and stamp
    //
    uint32_t hash = hashes[i];
    uint32_t iblock = hash >> bits_shift_for_block_and_stamp_;
    uint32_t stamp = iblock & stamp_mask;
    iblock >>= bits_shift_for_block_;

    const uint8_t* blockbase = block_data(iblock, num_block_bytes);
    ARROW_DCHECK(num_block_bytes % sizeof(uint64_t) == 0);
    uint64_t block = *reinterpret_cast<const uint64_t*>(blockbase);

    // Call helper functions to obtain the output triplet:
    // - match (of a stamp) found flag
    // - number of slots to skip before resuming further search, in case of no match or
    // false positive
    int match_found;
    int islot_in_block;
    search_block<false>(block, stamp, 0, &islot_in_block, &match_found);

    out_match_bitvector[i / 8] |= match_found << (i & 7);
    out_local_slots[i] = static_cast<uint8_t>(islot_in_block);
  }
}

// How many groups we can keep in the hash table without the need for resizing.
// When we reach this limit, we need to break processing of any further rows and resize.
//
int64_t SwissTable::num_groups_for_resize() const {
  // Consider N = 9 (aka 2 ^ 9 = 512 blocks) as small.
  // When N = 9, a slot id takes N + 3 = 12 bits, rounded up to 16 bits. This is also the
  // number of bits needed for a key id. Since each slot stores a status byte and a key
  // id, then a slot takes 1 byte + 16 bits = 3 bytes. Therefore a block of 8 slots takes
  // 24 bytes. The threshold of a small hash table ends up being 24 bytes * 512 = 12 KB.
  constexpr int log_blocks_small_ = 9;
  int64_t num_slots = num_slots_from_log_blocks(log_blocks_);
  if (log_blocks_ <= log_blocks_small_) {
    // Resize small hash tables when 50% full.
    return num_slots / 2;
  } else {
    // Resize large hash tables when 75% full.
    return num_slots * 3 / 4;
  }
}

uint32_t SwissTable::wrap_global_slot_id(uint32_t global_slot_id) const {
  uint32_t global_slot_id_mask =
      static_cast<uint32_t>((1ULL << (log_blocks_ + kLogSlotsPerBlock)) - 1ULL);
  return global_slot_id & global_slot_id_mask;
}

void SwissTable::early_filter(const int num_keys, const uint32_t* hashes,
                              uint8_t* out_match_bitvector,
                              uint8_t* out_local_slots) const {
  // Optimistically use simplified lookup involving only a start block to find
  // a single group id candidate for every input.
  int num_processed = 0;
#if defined(ARROW_HAVE_RUNTIME_AVX2) && defined(ARROW_HAVE_RUNTIME_BMI2)
  if ((hardware_flags_ & CpuInfo::AVX2) && CpuInfo::GetInstance()->HasEfficientBmi2()) {
    if (log_blocks_ <= 4) {
      num_processed = early_filter_imp_avx2_x32(num_keys, hashes, out_match_bitvector,
                                                out_local_slots);
    }
    num_processed += early_filter_imp_avx2_x8(
        num_keys - num_processed, hashes + num_processed,
        out_match_bitvector + num_processed / 8, out_local_slots + num_processed);
  }
#endif
  early_filter_imp(num_keys - num_processed, hashes + num_processed,
                   out_match_bitvector + num_processed / 8,
                   out_local_slots + num_processed);
}

// Input selection may be:
// - a range of all ids from 0 to num_keys - 1
// - a selection vector with list of ids
// - a bit-vector marking ids that are included
// Either selection index vector or selection bit-vector must be provided
// but both cannot be set at the same time (one must be null).
//
// Input and output selection index vectors are allowed to point to the same buffer
// (in-place filtering of ids).
//
// Output selection vector needs to have enough space for num_keys entries.
//
void SwissTable::run_comparisons(const int num_keys,
                                 const uint16_t* optional_selection_ids,
                                 const uint8_t* optional_selection_bitvector,
                                 const uint32_t* groupids, int* out_num_not_equal,
                                 uint16_t* out_not_equal_selection,
                                 const EqualImpl& equal_impl, void* callback_ctx) const {
  ARROW_DCHECK(optional_selection_ids || optional_selection_bitvector);
  ARROW_DCHECK(!optional_selection_ids || !optional_selection_bitvector);

  if (num_keys == 0) {
    *out_num_not_equal = 0;
    return;
  }

  if (!optional_selection_ids && optional_selection_bitvector) {
    // Count rows with matches (based on stamp comparison)
    // and decide based on their percentage whether to call dense or sparse comparison
    // function. Dense comparison means evaluating it for all inputs, even if the
    // matching stamp was not found. It may be cheaper to evaluate comparison for all
    // inputs if the extra cost of filtering is higher than the wasted processing of
    // rows with no match.
    //
    // Dense comparison can only be used if there is at least one inserted key,
    // because otherwise there is no key to compare to.
    //
    int64_t num_matches = arrow::internal::CountSetBits(optional_selection_bitvector,
                                                        /*offset=*/0, num_keys);

    if (num_inserted_ > 0 && num_matches > 0 && num_matches > 3 * num_keys / 4) {
      uint32_t out_num;
      equal_impl(num_keys, nullptr, groupids, &out_num, out_not_equal_selection,
                 callback_ctx);
      *out_num_not_equal = static_cast<int>(out_num);
    } else {
      util::bit_util::bits_to_indexes(1, hardware_flags_, num_keys,
                                      optional_selection_bitvector, out_num_not_equal,
                                      out_not_equal_selection);
      uint32_t out_num;
      equal_impl(*out_num_not_equal, out_not_equal_selection, groupids, &out_num,
                 out_not_equal_selection, callback_ctx);
      *out_num_not_equal = static_cast<int>(out_num);
    }
  } else {
    uint32_t out_num;
    equal_impl(num_keys, optional_selection_ids, groupids, &out_num,
               out_not_equal_selection, callback_ctx);
    *out_num_not_equal = static_cast<int>(out_num);
  }
}

// Given starting slot index, search blocks for a matching stamp
// until one is found or an empty slot is reached.
// If the search stopped on a non-empty slot, output corresponding
// group id from that slot.
//
// Return true if a match was found.
//
bool SwissTable::find_next_stamp_match(const uint32_t hash, const uint32_t in_slot_id,
                                       uint32_t* out_slot_id,
                                       uint32_t* out_group_id) const {
  const int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
  const int num_block_bytes = num_block_bytes_from_num_groupid_bits(num_groupid_bits);
  const int group_id_mask = group_id_mask_from_num_groupid_bits(num_groupid_bits);
  constexpr uint64_t stamp_mask = 0x7f;
  const int stamp =
      static_cast<int>((hash >> bits_shift_for_block_and_stamp_) & stamp_mask);
  uint32_t start_slot_id = wrap_global_slot_id(in_slot_id);
  int match_found;
  int local_slot;
  const uint8_t* blockbase;
  for (;;) {
    blockbase = block_data(start_slot_id >> kLogSlotsPerBlock, num_block_bytes);
    uint64_t block = *reinterpret_cast<const uint64_t*>(blockbase);

    search_block<true>(block, stamp, start_slot_id & kLocalSlotMask, &local_slot,
                       &match_found);

    start_slot_id =
        wrap_global_slot_id((start_slot_id & ~kLocalSlotMask) + local_slot + match_found);

    // Match found can be 1 in two cases:
    // - match was found
    // - match was not found in a full block
    // In the second case search needs to continue in the next block.
    if (match_found == 0 || blockbase[kMaxLocalSlot - local_slot] == stamp) {
      break;
    }
  }

  *out_group_id =
      extract_group_id(blockbase, local_slot, num_groupid_bits, group_id_mask);
  *out_slot_id = start_slot_id;

  return match_found;
}

// Find method is the continuation of processing from early_filter.
// Its input consists of hash values and the output of early_filter.
// It updates match bit-vector, clearing it from any false positives
// that might have been left by early_filter.
// It also outputs group ids, which are needed to be able to execute
// key comparisons. The caller may discard group ids if only the
// match flag is of interest.
//
void SwissTable::find(const int num_keys, const uint32_t* hashes,
                      uint8_t* inout_match_bitvector, const uint8_t* local_slots,
                      uint32_t* out_group_ids, util::TempVectorStack* temp_stack,
                      const EqualImpl& equal_impl, void* callback_ctx) const {
  // Temporary selection vector.
  // It will hold ids of keys for which we do not know yet
  // if they have a match in hash table or not.
  //
  // Initially the set of these keys is represented by input
  // match bit-vector. Eventually we switch from this bit-vector
  // to array of ids.
  //
  ARROW_DCHECK(num_keys <= (1 << log_minibatch_));
  auto ids_buf = util::TempVectorHolder<uint16_t>(temp_stack, num_keys);
  uint16_t* ids = ids_buf.mutable_data();
  int num_ids;

  int64_t num_matches = arrow::internal::CountSetBits(inout_match_bitvector,
                                                      /*offset=*/0, num_keys);

  // If there is a high density of selected input rows
  // (majority of them are present in the selection),
  // we may run some computation on all of the input rows ignoring
  // selection and then filter the output of this computation
  // (pre-filtering vs post-filtering).
  //
  bool visit_all = num_matches > 0 && num_matches > 3 * num_keys / 4;
  if (visit_all) {
    extract_group_ids(num_keys, nullptr, hashes, local_slots, out_group_ids);
    run_comparisons(num_keys, nullptr, inout_match_bitvector, out_group_ids, &num_ids,
                    ids, equal_impl, callback_ctx);
  } else {
    util::bit_util::bits_to_indexes(1, hardware_flags_, num_keys, inout_match_bitvector,
                                    &num_ids, ids);
    extract_group_ids(num_ids, ids, hashes, local_slots, out_group_ids);
    run_comparisons(num_ids, ids, nullptr, out_group_ids, &num_ids, ids, equal_impl,
                    callback_ctx);
  }

  if (num_ids == 0) {
    return;
  }

  auto slot_ids_buf = util::TempVectorHolder<uint32_t>(temp_stack, num_keys);
  uint32_t* slot_ids = slot_ids_buf.mutable_data();
  init_slot_ids(num_ids, ids, hashes, local_slots, inout_match_bitvector, slot_ids);

  while (num_ids > 0) {
    int num_ids_last_iteration = num_ids;
    num_ids = 0;
    for (int i = 0; i < num_ids_last_iteration; ++i) {
      int id = ids[i];
      uint32_t next_slot_id;
      bool match_found = find_next_stamp_match(hashes[id], slot_ids[id], &next_slot_id,
                                               &(out_group_ids[id]));
      slot_ids[id] = next_slot_id;
      // If next match was not found then clear match bit in a bit vector
      if (!match_found) {
        ::arrow::bit_util::ClearBit(inout_match_bitvector, id);
      } else {
        ids[num_ids++] = id;
      }
    }

    run_comparisons(num_ids, ids, nullptr, out_group_ids, &num_ids, ids, equal_impl,
                    callback_ctx);
  }
}

// Slow processing of input keys in the most generic case.
// Handles inserting new keys.
// Preexisting keys will be handled correctly, although the intended use is for this
// call to follow a call to find() method, which would only pass on new keys that were
// not present in the hash table.
//
// Probe selected keys, compare stamp matches, then commit new keys in input order.
// Update selection vector to keep items that were not processed due to resize.
// Ids in selection vector do not have to be sorted.
//
// 中文说明：这里最核心的问题是“group id 的分配顺序”必须跟输入中第一次
// 看到 key 的顺序一致。探测哈希表时，后面的 key 可能先遇到空槽；如果马上
// 写入，就可能越过前面仍在等待相等性比较的 key，导致 uniques 顺序被打乱。
//
Status SwissTable::map_new_keys_helper(
    const uint32_t* hashes, uint32_t* inout_num_selected, uint16_t* inout_selection,
    bool* out_need_resize, uint32_t* out_group_ids, uint32_t* inout_next_slot_ids,
    util::TempVectorStack* temp_stack, const EqualImpl& equal_impl,
    const AppendImpl& append_impl, void* callback_ctx) {
  auto num_groups_limit = num_groups_for_resize();
  ARROW_DCHECK(num_inserted_ < num_groups_limit);

  // Temporary arrays are of limited size.
  // The input needs to be split into smaller portions if it exceeds that limit.
  //
  ARROW_DCHECK(*inout_num_selected <= static_cast<uint32_t>(1 << log_minibatch_));

  // Check the alignment of the input selection vector
  ARROW_DCHECK((reinterpret_cast<uint64_t>(inout_selection) & 1) == 0);

  // 中文说明：只需要保存本轮会追加的新 key 的 row id，容量取 selection 大小。
  auto append_ids_buffer =
      util::TempVectorHolder<uint16_t>(temp_stack, *inout_num_selected);
  // 中文说明：append_impl 接收裸指针，这里取出临时缓冲的可写地址。
  uint16_t* append_ids = append_ids_buffer.mutable_data();
  // 中文说明：记录当前攒了多少个尚未真正写入 rows_ 的新 group。
  uint32_t num_pending_appends = 0;

  // Keep appends batched for the common all-new path. Flush only when a later
  // comparison needs to read a group inserted earlier in this helper call.
  // 中文说明：按输入顺序先给新 key 分配 group id；row 内容仍批量追加，只有
  // 比较到本轮刚插入的 group 时才 flush，避免 all-new 路径退化成逐行 append。
  auto flush_appends = [&]() -> Status {
    // 中文说明：没有 pending append 时直接返回，避免空 append 调用。
    if (num_pending_appends == 0) {
      // 中文说明：空 flush 是正常路径，不代表错误。
      return Status::OK();
    }
    // 中文说明：把 pending row 真正复制到 rows_，让后续比较能读到 key 内容。
    RETURN_NOT_OK(append_impl(num_pending_appends, append_ids, callback_ctx));
    // 中文说明：这些 pending row 已经成为正式 group，全局 inserted 计数前移。
    num_inserted_ += num_pending_appends;
    // 中文说明：清空 pending 数量，后续继续复用 append_ids 缓冲。
    num_pending_appends = 0;
    // 中文说明：flush 成功完成。
    return Status::OK();
  };

  // 中文说明：记录 selection 前缀里已经完成判定的 row 数量。
  uint32_t num_processed = 0;
  // 中文说明：第一性原理版本只保留一个顺序循环，严格按 selection 顺序决定每个 key。
  for (; num_processed < *inout_num_selected;) {
    // 中文说明：selection 里保存的是原始 minibatch row id。
    int id = inout_selection[num_processed];

    // 中文说明：同一个 key 可能连续遇到多个 stamp 相同但内容不同的候选 group。
    for (;;) {
      // 中文说明：沿 probing 链查找下一个 stamp 匹配的候选槽；找不到就是新 key。
      bool match_found =
          find_next_stamp_match(hashes[id], inout_next_slot_ids[id],
                                &inout_next_slot_ids[id], &out_group_ids[id]);
      // 中文说明：没有候选 match 时，当前 key 在当前位置按输入顺序创建新 group。
      if (!match_found) {
        // 中文说明：新 group id 等于已落盘 group 数加上本轮 pending group 数。
        out_group_ids[id] = num_inserted_ + num_pending_appends;
        // 中文说明：把新 group id 写入当前空槽，让后续同 batch 重复 key 能找到它。
        insert_into_empty_slot(inout_next_slot_ids[id], hashes[id], out_group_ids[id]);
        // 中文说明：保存完整 hash，供后续 probing、比较和 grow 迁移使用。
        this->hashes()[inout_next_slot_ids[id]] = hashes[id];
        // 中文说明：row 内容先记录到 pending append，等待批量写入 rows_。
        append_ids[num_pending_appends++] = static_cast<uint16_t>(id);
        // 中文说明：当前 selection 项已经完成判定，前缀长度前移一位。
        ++num_processed;

        // 中文说明：如果这个新 group 让表达到 resize 阈值，本轮到此停止。
        if (num_inserted_ + num_pending_appends == num_groups_limit) {
          // 中文说明：跳出当前 key 的 probing 循环，外层会保留剩余 selection。
          break;
        }
        // 中文说明：当前 key 已作为新 group 处理完，继续下一个 selection 项。
        break;
      }

      // If the matched group was inserted in this helper call, make its row visible
      // before comparing. Existing groups are already present in rows_.
      if (out_group_ids[id] >= num_inserted_) {
        // 中文说明：候选 group 属于本轮 pending append，先 flush 才能比较真实 row。
        RETURN_NOT_OK(flush_appends());
      }

      // 中文说明：run_comparisons 接收 row id 数组，这里构造单元素输入。
      uint16_t temp_id = static_cast<uint16_t>(id);
      // 中文说明：接收“不相等”的数量；0 表示当前 key 命中已有 group。
      int num_not_equal;
      // 中文说明：只比较当前 row 和刚找到的候选 group。
      run_comparisons(1, &temp_id, nullptr, out_group_ids, &num_not_equal, &temp_id,
                      equal_impl, callback_ctx);
      // 中文说明：比较相等时，out_group_ids[id] 已经是正确 group id。
      if (num_not_equal == 0) {
        // 中文说明：当前 selection 项已经完成判定，继续下一个输入位置。
        ++num_processed;
        // 中文说明：退出当前 key 的 probing 循环。
        break;
      }
    }

    // 中文说明：容量满时不能继续创建新 group，剩余 key 交给扩容后下一轮处理。
    if (num_inserted_ + num_pending_appends == num_groups_limit) {
      break;
    }
  }
  // 中文说明：本轮结束前把还没落到 rows_ 的新 group 一次性写入。
  RETURN_NOT_OK(flush_appends());

  // 中文说明：如果因为容量限制中断，本轮没处理完的 selection 要搬到前面。
  // 调用方扩容后会继续处理这些剩余 key。
  if (num_processed < *inout_num_selected) {
    // 中文说明：只保留未处理的 suffix，调用方扩容后会继续处理这些 row。
    memmove(inout_selection, inout_selection + num_processed,
            sizeof(uint16_t) * (*inout_num_selected - num_processed));
  }
  // 中文说明：更新 selection 长度为“还没有处理完”的数量。
  *inout_num_selected = *inout_num_selected - num_processed;

  *out_need_resize = (num_inserted_ == num_groups_limit);
  return Status::OK();
}

// Do inserts and find group ids for a set of new keys (with possible duplicates within
// this set).
//
Status SwissTable::map_new_keys(uint32_t num_ids, uint16_t* ids, const uint32_t* hashes,
                                uint32_t* group_ids, util::TempVectorStack* temp_stack,
                                const EqualImpl& equal_impl,
                                const AppendImpl& append_impl, void* callback_ctx) {
  if (num_ids == 0) {
    return Status::OK();
  }

  uint16_t max_id = ids[0];
  for (uint32_t i = 1; i < num_ids; ++i) {
    max_id = std::max(max_id, ids[i]);
  }

  // Temporary buffers have limited size.
  // Caller is responsible for splitting larger input arrays into smaller chunks.
  ARROW_DCHECK(static_cast<int>(num_ids) <= (1 << log_minibatch_));
  ARROW_DCHECK(static_cast<int>(max_id + 1) <= (1 << log_minibatch_));

  // Allocate temporary buffers for slot ids and initialize them
  auto slot_ids_buf = util::TempVectorHolder<uint32_t>(temp_stack, max_id + 1);
  uint32_t* slot_ids = slot_ids_buf.mutable_data();
  init_slot_ids_for_new_keys(num_ids, ids, hashes, slot_ids);

  do {
    // A single round of slow-pass (robust) lookup or insert.
    // A single round ends with either a single comparison verifying the match
    // candidate or inserting a new key. A single round of slow-pass may return early
    // if we reach the limit of the number of groups due to inserts of new keys. In
    // that case we need to resize and recalculating starting global slot ids for new
    // bigger hash table.
    bool out_of_capacity;
    RETURN_NOT_OK(map_new_keys_helper(hashes, &num_ids, ids, &out_of_capacity, group_ids,
                                      slot_ids, temp_stack, equal_impl, append_impl,
                                      callback_ctx));
    if (out_of_capacity) {
      RETURN_NOT_OK(grow_double());
      // Reset start slot ids for still unprocessed input keys.
      //
      for (uint32_t i = 0; i < num_ids; ++i) {
        // First slot in the new starting block
        const int16_t id = ids[i];
        uint32_t block_id = block_id_from_hash(hashes[id], log_blocks_);
        slot_ids[id] = global_slot_id(block_id, /*local_slot_id=*/0);
      }
    }
  } while (num_ids > 0);

  return Status::OK();
}

Status SwissTable::grow_double() {
  // Before and after metadata
  int num_group_id_bits_before = num_groupid_bits_from_log_blocks(log_blocks_);
  int num_group_id_bits_after = num_groupid_bits_from_log_blocks(log_blocks_ + 1);
  uint32_t group_id_mask_before =
      group_id_mask_from_num_groupid_bits(num_group_id_bits_before);
  int log_blocks_after = log_blocks_ + 1;
  int bits_shift_for_block_and_stamp_after =
      ComputeBitsShiftForBlockAndStamp(log_blocks_after);
  int bits_shift_for_block_after = ComputeBitsShiftForBlock(log_blocks_after);
  int block_size_before = num_block_bytes_from_num_groupid_bits(num_group_id_bits_before);
  int block_size_after = num_block_bytes_from_num_groupid_bits(num_group_id_bits_after);
  int64_t block_size_total_after =
      num_bytes_total_blocks(block_size_after, log_blocks_after);
  int64_t hashes_size_total_after =
      (bits_hash_ / 8 * num_slots_from_log_blocks(log_blocks_after)) + padding_;
  constexpr uint32_t stamp_mask = (1 << bits_stamp_) - 1;

  // Allocate new buffers
  ARROW_ASSIGN_OR_RAISE(std::unique_ptr<Buffer> blocks_new,
                        AllocateBuffer(block_size_total_after, pool_));
  memset(blocks_new->mutable_data(), 0, block_size_total_after);
  ARROW_ASSIGN_OR_RAISE(std::unique_ptr<Buffer> hashes_new_buffer,
                        AllocateBuffer(hashes_size_total_after, pool_));
  auto hashes_new = reinterpret_cast<uint32_t*>(hashes_new_buffer->mutable_data());

  // First pass over all old blocks.
  // Reinsert entries that were not in the overflow block
  // (block other than selected by hash bits corresponding to the entry).
  for (int i = 0; i < (1 << log_blocks_); ++i) {
    // How many full slots in this block
    const uint8_t* block_base = block_data(i, block_size_before);
    uint8_t* double_block_base_new =
        mutable_block_data(blocks_new->mutable_data(), 2 * i, block_size_after);
    uint64_t block = *reinterpret_cast<const uint64_t*>(block_base);

    uint32_t full_slots = std::countl_zero(block & kHighBitOfEachByte) >> 3;
    uint32_t full_slots_new[2];
    full_slots_new[0] = full_slots_new[1] = 0;
    util::SafeStore(double_block_base_new, kHighBitOfEachByte);
    util::SafeStore(double_block_base_new + block_size_after, kHighBitOfEachByte);

    for (uint32_t j = 0; j < full_slots; ++j) {
      uint32_t slot_id = global_slot_id(i, j);
      uint32_t hash = hashes()[slot_id];
      uint32_t block_id_new = block_id_from_hash(hash, log_blocks_after);
      bool is_overflow_entry = ((block_id_new >> 1) != static_cast<uint64_t>(i));
      if (is_overflow_entry) {
        continue;
      }

      uint32_t ihalf = block_id_new & 1;
      uint8_t stamp_new = (hash >> bits_shift_for_block_and_stamp_after) & stamp_mask;
      uint32_t group_id =
          extract_group_id(block_base, j, num_group_id_bits_before, group_id_mask_before);
      uint32_t slot_id_new = global_slot_id(i * 2 + ihalf, full_slots_new[ihalf]);
      hashes_new[slot_id_new] = hash;
      uint8_t* block_base_new = double_block_base_new + ihalf * block_size_after;
      block_base_new[kMaxLocalSlot - full_slots_new[ihalf]] = stamp_new;
      int group_id_bit_offs_new = full_slots_new[ihalf] * num_group_id_bits_after;
      uint64_t* ptr = reinterpret_cast<uint64_t*>(
          block_base_new + bytes_status_in_block_ + (group_id_bit_offs_new >> 3));
      util::SafeStore(ptr, util::SafeLoad(ptr) | (static_cast<uint64_t>(group_id)
                                                  << (group_id_bit_offs_new & 7)));
      full_slots_new[ihalf]++;
    }
  }

  // Second pass over all old blocks.
  // Reinsert entries that were in an overflow block.
  for (int i = 0; i < (1 << log_blocks_); ++i) {
    // How many full slots in this block
    const uint8_t* block_base = block_data(i, block_size_before);
    uint64_t block = util::SafeLoadAs<uint64_t>(block_base);
    uint32_t full_slots = std::countl_zero(block & kHighBitOfEachByte) >> 3;

    for (uint32_t j = 0; j < full_slots; ++j) {
      uint32_t slot_id = global_slot_id(i, j);
      uint32_t hash = hashes()[slot_id];
      uint32_t block_id_new = block_id_from_hash(hash, log_blocks_after);
      bool is_overflow_entry = ((block_id_new >> 1) != static_cast<uint64_t>(i));
      if (!is_overflow_entry) {
        continue;
      }

      uint32_t group_id =
          extract_group_id(block_base, j, num_group_id_bits_before, group_id_mask_before);
      uint8_t stamp_new = (hash >> bits_shift_for_block_and_stamp_after) & stamp_mask;

      uint8_t* block_base_new =
          mutable_block_data(blocks_new->mutable_data(), block_id_new, block_size_after);
      uint64_t block_new = util::SafeLoadAs<uint64_t>(block_base_new);
      int full_slots_new =
          static_cast<int>(std::countl_zero(block_new & kHighBitOfEachByte) >> 3);
      while (full_slots_new == kSlotsPerBlock) {
        block_id_new = (block_id_new + 1) & ((1 << log_blocks_after) - 1);
        block_base_new = blocks_new->mutable_data() + block_id_new * block_size_after;
        block_new = util::SafeLoadAs<uint64_t>(block_base_new);
        full_slots_new =
            static_cast<int>(std::countl_zero(block_new & kHighBitOfEachByte) >> 3);
      }

      hashes_new[block_id_new * kSlotsPerBlock + full_slots_new] = hash;
      block_base_new[kMaxLocalSlot - full_slots_new] = stamp_new;
      int group_id_bit_offs_new = full_slots_new * num_group_id_bits_after;
      uint64_t* ptr = reinterpret_cast<uint64_t*>(
          block_base_new + bytes_status_in_block_ + (group_id_bit_offs_new >> 3));
      util::SafeStore(ptr, util::SafeLoad(ptr) | (static_cast<uint64_t>(group_id)
                                                  << (group_id_bit_offs_new & 7)));
    }
  }

  blocks_ = std::move(blocks_new);
  hashes_ = std::move(hashes_new_buffer);
  log_blocks_ = log_blocks_after;
  bits_shift_for_block_and_stamp_ = bits_shift_for_block_and_stamp_after;
  bits_shift_for_block_ = bits_shift_for_block_after;

  return Status::OK();
}

Status SwissTable::init(int64_t hardware_flags, MemoryPool* pool, int log_blocks,
                        bool no_hash_array) {
  hardware_flags_ = hardware_flags;
  pool_ = pool;
  log_minibatch_ = util::MiniBatch::kLogMiniBatchLength;

  log_blocks_ = log_blocks;
  bits_shift_for_block_and_stamp_ = ComputeBitsShiftForBlockAndStamp(log_blocks_);
  bits_shift_for_block_ = ComputeBitsShiftForBlock(log_blocks_);
  int num_groupid_bits = num_groupid_bits_from_log_blocks(log_blocks_);
  num_inserted_ = 0;

  const int block_bytes = num_block_bytes_from_num_groupid_bits(num_groupid_bits);
  const int64_t slot_bytes = num_bytes_total_blocks(block_bytes, log_blocks_);
  ARROW_ASSIGN_OR_RAISE(blocks_, AllocateBuffer(slot_bytes, pool_));

  // Make sure group ids are initially set to zero for all slots.
  memset(blocks_->mutable_data(), 0, slot_bytes);

  // Initialize all status bytes to represent an empty slot.
  for (int i = 0; i < 1 << log_blocks_; ++i) {
    auto block = mutable_block_data(i, block_bytes);
    util::SafeStore(block, kHighBitOfEachByte);
  }

  if (no_hash_array) {
    hashes_ = nullptr;
  } else {
    int64_t num_slots = num_slots_from_log_blocks(log_blocks);
    const int hash_size = bits_hash_ >> 3;
    const int64_t hash_bytes = hash_size * num_slots + padding_;
    ARROW_ASSIGN_OR_RAISE(hashes_, AllocateBuffer(hash_bytes, pool_));
  }

  return Status::OK();
}

void SwissTable::cleanup() {
  if (blocks_) {
    blocks_ = nullptr;
  }
  if (hashes_) {
    hashes_ = nullptr;
  }
  log_blocks_ = 0;
  bits_shift_for_block_and_stamp_ = ComputeBitsShiftForBlockAndStamp(log_blocks_);
  bits_shift_for_block_ = ComputeBitsShiftForBlock(log_blocks_);
  num_inserted_ = 0;
}

}  // namespace compute
}  // namespace arrow
