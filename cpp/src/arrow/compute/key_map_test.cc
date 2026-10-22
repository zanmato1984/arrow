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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "arrow/compute/key_map_internal.h"
#include "arrow/memory_pool.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/util.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/cpu_info.h"

namespace arrow {

using internal::CpuInfo;

namespace compute {

namespace {

void CheckPlacementSeed(SwissTable* table, std::optional<uint32_t> seed,
                        int64_t hardware_flags, bool collide,
                        std::vector<uint32_t>* stored_hashes) {
  const uint32_t num_keys = collide ? 257 : 70001;
  auto original_hash = [collide](uint32_t key) {
    return collide ? 0U : key * 2654435761U;
  };
  table->cleanup();
  ASSERT_OK(table->init(hardware_flags, default_memory_pool(), /*log_blocks=*/0,
                        /*no_hash_array=*/false, seed));
  ASSERT_EQ(table->hash_seed(), seed);
  util::TempVectorStack stack;
  ASSERT_OK(stack.Init(default_memory_pool(), 64 * table->minibatch_size()));
  std::vector<uint32_t> current_keys, group_keys;
  std::vector<uint32_t> expected_ids(num_keys);
  SwissTable::EqualImpl equal = [&](int count, const uint16_t* selection,
                                    const uint32_t* ids, uint32_t* num_mismatch,
                                    uint16_t* mismatch, void*) {
    *num_mismatch = 0;
    for (int i = 0; i < count; ++i) {
      uint16_t row = selection ? selection[i] : static_cast<uint16_t>(i);
      if (ids[row] >= group_keys.size() || current_keys[row] != group_keys[ids[row]]) {
        mismatch[(*num_mismatch)++] = row;
      }
    }
  };
  SwissTable::AppendImpl append = [&](int count, const uint16_t* selection, void*) {
    for (int i = 0; i < count; ++i) group_keys.push_back(current_keys[selection[i]]);
    return Status::OK();
  };
  auto process = [&](bool insert) {
    int count = static_cast<int>(current_keys.size());
    std::vector<uint32_t> hashes(count), ids(count);
    std::vector<uint8_t> match((count + 7) / 8 + 64), slots(count + 64);
    for (int i = 0; i < count; ++i) hashes[i] = original_hash(current_keys[i]);
    const auto original = hashes;
    table->early_filter(count, hashes.data(), match.data(), slots.data());
    table->find(count, hashes.data(), match.data(), slots.data(), ids.data(), &stack,
                equal, nullptr);
    if (insert) {
      std::vector<uint16_t> selection;
      for (int i = 0; i < count; ++i) {
        if (!bit_util::GetBit(match.data(), i)) selection.push_back(i);
      }
      ASSERT_OK(table->map_new_keys(static_cast<uint32_t>(selection.size()),
                                    selection.data(), hashes.data(), ids.data(), &stack,
                                    equal, append, nullptr));
    }
    ASSERT_EQ(hashes, original);  // The table never rewrites the caller's hashes.
    for (int i = 0; i < count; ++i) {
      auto key = current_keys[i];
      if (insert || key < num_keys) {
        if (!insert) ASSERT_TRUE(bit_util::GetBit(match.data(), i));
        ASSERT_LT(ids[i], group_keys.size());
        ASSERT_EQ(group_keys[ids[i]], key);
        if (insert)
          expected_ids[key] = ids[i];
        else
          ASSERT_EQ(ids[i], expected_ids[key]);
      } else {
        ASSERT_FALSE(bit_util::GetBit(match.data(), i));
      }
    }
  };
  // Cover both AVX2 batch sizes, scalar tails and growth through all group-id
  // widths, including growth while a sparse selection still needs inserting.
  constexpr std::array<int, 11> sizes = {1, 7, 8, 9, 31, 32, 33, 127, 511, 1023, 1024};
  uint32_t offset = 0;
  for (int batch = 0; offset < num_keys; ++batch) {
    const uint32_t capacity = sizes[batch % sizes.size()];
    const bool include_existing = offset > 0 && capacity > 1;
    auto count = std::min<uint32_t>(capacity - include_existing, num_keys - offset);
    current_keys.resize(count + include_existing);
    uint32_t next = offset;
    for (size_t i = 0; i < current_keys.size(); ++i) {
      current_keys[i] = include_existing && i == count / 2 ? 0 : next++;
    }
    process(true);
    offset += count;
  }
  ASSERT_EQ(table->num_inserted(), num_keys);
  for (offset = 0; offset < num_keys; offset += 1023) {
    auto count = std::min<uint32_t>(1023, num_keys - offset);
    current_keys.resize(count);
    for (uint32_t i = 0; i < count; ++i) current_keys[i] = num_keys - 1 - offset - i;
    process(false);
  }
  current_keys = {0, num_keys, 0, num_keys + 1, num_keys - 1};
  process(false);  // Repeated existing keys mixed with absent keys.
  current_keys = {0, 1, 0, 1};
  process(true);
  ASSERT_EQ(table->num_inserted(), num_keys);

  stored_hashes->assign(num_keys, 0);
  int bits = SwissTable::num_groupid_bits_from_log_blocks(table->log_blocks());
  int bytes = SwissTable::num_block_bytes_from_num_groupid_bits(bits);
  for (uint32_t b = 0; b < (1U << table->log_blocks()); ++b) {
    const uint8_t* block = table->block_data(b, bytes);
    for (int slot = 0; slot < SwissTable::kSlotsPerBlock; ++slot) {
      if (block[7 - slot] & 0x80) break;
      auto id = SwissTable::extract_group_id(
          block, slot, bits, SwissTable::group_id_mask_from_num_groupid_bits(bits));
      ASSERT_LT(id, group_keys.size());
      (*stored_hashes)[group_keys[id]] =
          table->hashes()[SwissTable::global_slot_id(b, slot)];
      if (!seed) {
        ASSERT_EQ((*stored_hashes)[group_keys[id]], original_hash(group_keys[id]));
      }
    }
  }
}

}  // namespace

TEST(SwissTable, PlacementSeedsAndGrowth) {
  for (int64_t flags : GetSupportedHardwareFlags({CpuInfo::AVX2})) {
    SwissTable table;
    std::vector<uint32_t> original, zero, one, again;
    CheckPlacementSeed(&table, std::nullopt, flags, false, &original);
    CheckPlacementSeed(&table, 0, flags, false, &zero);
    CheckPlacementSeed(&table, 1, flags, false, &one);
    ASSERT_NE(original, zero);
    ASSERT_NE(zero, one);
    CheckPlacementSeed(&table, 0xffffffffU, flags, false, &again);
    ASSERT_NE(one, again);
    CheckPlacementSeed(&table, 0, flags, false, &again);
    ASSERT_EQ(zero, again);  // Reinitializing with the same seed is reproducible.
    CheckPlacementSeed(&table, std::nullopt, flags, false, &again);
    ASSERT_EQ(original, again);  // Reinitializing without a seed restores the default.
  }
}

TEST(SwissTable, PlacementSeedWithFullHashCollisions) {
  for (int64_t flags : GetSupportedHardwareFlags({CpuInfo::AVX2})) {
    SwissTable table;
    std::vector<uint32_t> stored;
    for (auto seed : {0U, 1U, 0xffffffffU}) {
      CheckPlacementSeed(&table, seed, flags, true, &stored);
    }
  }
}

// With 32-bit group ids a block takes 40 bytes, so the byte offsets of the blocks from
// id ceil(2^32 / 40) on do not fit in 32 bits.
TEST(SwissTable, LARGE_MEMORY_TEST(EarlyFilterOver4GB)) {
  if constexpr (sizeof(void*) == 4) {
    GTEST_SKIP() << "Test only works on 64-bit platforms";
  }

  // 2^27 blocks of 40 bytes take 5GB.
  constexpr int kLogBlocks = 27;
  constexpr uint32_t kFirstBlockOver4GB = 107374183;
  constexpr int kNumHashes = 16;

  // One hash for each of the 8 blocks below and the 8 blocks above the 4GB boundary,
  // all with a non-zero stamp.
  std::vector<uint32_t> hashes(kNumHashes);
  for (int i = 0; i < kNumHashes; ++i) {
    uint32_t block_id = kFirstBlockOver4GB - kNumHashes / 2 + i;
    hashes[i] = (block_id << (SwissTable::bits_hash_ - kLogBlocks)) | 1;
  }

  for (int64_t hardware_flags : GetSupportedHardwareFlags({CpuInfo::AVX2})) {
    ARROW_SCOPED_TRACE("hardware_flags = ", hardware_flags);
    SwissTable table;
    ASSERT_OK(table.init(hardware_flags, default_memory_pool(), kLogBlocks,
                         /*no_hash_array=*/true));
    // Insert every other hash into the first slot of its block, leaving the other blocks
    // empty.
    for (int i = 0; i < kNumHashes; i += 2) {
      uint32_t block_id = SwissTable::block_id_from_hash(hashes[i], kLogBlocks);
      table.insert_into_empty_slot(SwissTable::global_slot_id(block_id, 0), hashes[i],
                                   /*group_id=*/i);
    }

    uint8_t match_bitvector[kNumHashes / 8];
    uint8_t local_slots[kNumHashes];
    table.early_filter(kNumHashes, hashes.data(), match_bitvector, local_slots);
    for (int i = 0; i < kNumHashes; ++i) {
      ARROW_SCOPED_TRACE("block_id = ",
                         SwissTable::block_id_from_hash(hashes[i], kLogBlocks));
      // Inserted hashes match in the first slot, the others hit an empty block whose
      // first slot is empty.
      ASSERT_EQ(bit_util::GetBit(match_bitvector, i), i % 2 == 0);
      ASSERT_EQ(local_slots[i], 0);
    }
  }
}

// When growing to a table over 4GB, entries that have to move past a full block must
// still land in the right block.
TEST(SwissTable, LARGE_MEMORY_TEST(GrowOver4GB)) {
  if constexpr (sizeof(void*) == 4) {
    GTEST_SKIP() << "Test only works on 64-bit platforms";
  }

  // Grow from 2^26 to 2^27 blocks.
  constexpr int kLogBlocks = 26;
  constexpr uint32_t kBlockId = (1u << kLogBlocks) - 2;
  constexpr int kNumHashes = SwissTable::kSlotsPerBlock + 1;

  // All these hashes map to block kBlockId before growing and to block 2 * kBlockId
  // after. The first 8 fill these blocks, so the last one is in block kBlockId + 1 before
  // growing and has to move to block 2 * kBlockId + 1, which is over 4GB.
  std::vector<uint32_t> hashes(kNumHashes);
  for (int i = 0; i < kNumHashes; ++i) {
    hashes[i] = (kBlockId << (SwissTable::bits_hash_ - kLogBlocks)) | i;
  }

  // Key i is equal to group id i.
  SwissTable::EqualImpl equal_impl =
      [](int num_keys, const uint16_t* selection, const uint32_t* group_ids,
         uint32_t* out_num_keys_mismatch, uint16_t* out_selection_mismatch, void*) {
        *out_num_keys_mismatch = 0;
        for (int i = 0; i < num_keys; ++i) {
          uint16_t id = selection ? selection[i] : static_cast<uint16_t>(i);
          if (group_ids[id] != id) {
            out_selection_mismatch[(*out_num_keys_mismatch)++] = id;
          }
        }
      };
  SwissTable::AppendImpl append_impl = [](int, const uint16_t*, void*) {
    return Status::OK();
  };

  for (int64_t hardware_flags : GetSupportedHardwareFlags({CpuInfo::AVX2})) {
    ARROW_SCOPED_TRACE("hardware_flags = ", hardware_flags);
    SwissTable table;
    ASSERT_OK(table.init(hardware_flags, default_memory_pool(), kLogBlocks));
    for (int i = 0; i < kNumHashes; ++i) {
      uint32_t slot_id = SwissTable::global_slot_id(kBlockId, 0) + i;
      table.insert_into_empty_slot(slot_id, hashes[i], /*group_id=*/i);
      table.hashes()[slot_id] = hashes[i];
    }

    // The table grows when 75% of its slots are used. Pretend that it is one key short
    // of that and insert one more key.
    table.num_inserted(
        static_cast<uint32_t>((int64_t{1} << (kLogBlocks + 3)) * 3 / 4 - 1));
    util::TempVectorStack temp_stack;
    ASSERT_OK(temp_stack.Init(default_memory_pool(), 64 * table.minibatch_size()));
    uint16_t new_key_id = 0;
    uint32_t new_key_hash = 0;
    uint32_t new_group_id;
    ASSERT_OK(table.map_new_keys(/*num_ids=*/1, &new_key_id, &new_key_hash, &new_group_id,
                                 &temp_stack, equal_impl, append_impl,
                                 /*callback_ctx=*/nullptr));
    ASSERT_EQ(table.log_blocks(), kLogBlocks + 1);

    uint8_t match_bitvector[(kNumHashes + 7) / 8];
    uint8_t local_slots[kNumHashes];
    uint32_t group_ids[kNumHashes];
    table.early_filter(kNumHashes, hashes.data(), match_bitvector, local_slots);
    table.find(kNumHashes, hashes.data(), match_bitvector, local_slots, group_ids,
               &temp_stack, equal_impl, /*callback_ctx=*/nullptr);
    for (int i = 0; i < kNumHashes; ++i) {
      ARROW_SCOPED_TRACE("key = ", i);
      ASSERT_TRUE(bit_util::GetBit(match_bitvector, i));
      ASSERT_EQ(group_ids[i], i);
    }
  }
}

}  // namespace compute
}  // namespace arrow
