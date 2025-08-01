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

#include "parquet/arrow/schema_internal.h"

#include "arrow/extension/json.h"
#include "arrow/extension/uuid.h"
#include "arrow/type.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/logging.h"
#include "arrow/util/string.h"

#include "parquet/geospatial/util_json_internal.h"
#include "parquet/properties.h"

using ArrowType = ::arrow::DataType;
using ArrowTypeId = ::arrow::Type;
using ParquetType = parquet::Type;

namespace parquet::arrow {

using ::arrow::Result;
using ::arrow::Status;
using ::arrow::internal::checked_cast;

namespace {

Result<std::shared_ptr<ArrowType>> MakeArrowDecimal(const LogicalType& logical_type) {
  const auto& decimal = checked_cast<const DecimalLogicalType&>(logical_type);
  if (decimal.precision() <= ::arrow::Decimal128Type::kMaxPrecision) {
    return ::arrow::Decimal128Type::Make(decimal.precision(), decimal.scale());
  }
  return ::arrow::Decimal256Type::Make(decimal.precision(), decimal.scale());
}

Result<std::shared_ptr<ArrowType>> MakeArrowInt(const LogicalType& logical_type) {
  const auto& integer = checked_cast<const IntLogicalType&>(logical_type);
  switch (integer.bit_width()) {
    case 8:
      return integer.is_signed() ? ::arrow::int8() : ::arrow::uint8();
    case 16:
      return integer.is_signed() ? ::arrow::int16() : ::arrow::uint16();
    case 32:
      return integer.is_signed() ? ::arrow::int32() : ::arrow::uint32();
    default:
      return Status::TypeError(logical_type.ToString(),
                               " cannot annotate physical type Int32");
  }
}

Result<std::shared_ptr<ArrowType>> MakeArrowInt64(const LogicalType& logical_type) {
  const auto& integer = checked_cast<const IntLogicalType&>(logical_type);
  switch (integer.bit_width()) {
    case 64:
      return integer.is_signed() ? ::arrow::int64() : ::arrow::uint64();
    default:
      return Status::TypeError(logical_type.ToString(),
                               " cannot annotate physical type Int64");
  }
}

Result<std::shared_ptr<ArrowType>> MakeArrowTime32(const LogicalType& logical_type) {
  const auto& time = checked_cast<const TimeLogicalType&>(logical_type);
  switch (time.time_unit()) {
    case LogicalType::TimeUnit::MILLIS:
      return ::arrow::time32(::arrow::TimeUnit::MILLI);
    default:
      return Status::TypeError(logical_type.ToString(),
                               " cannot annotate physical type Time32");
  }
}

Result<std::shared_ptr<ArrowType>> MakeArrowTime64(const LogicalType& logical_type) {
  const auto& time = checked_cast<const TimeLogicalType&>(logical_type);
  switch (time.time_unit()) {
    case LogicalType::TimeUnit::MICROS:
      return ::arrow::time64(::arrow::TimeUnit::MICRO);
    case LogicalType::TimeUnit::NANOS:
      return ::arrow::time64(::arrow::TimeUnit::NANO);
    default:
      return Status::TypeError(logical_type.ToString(),
                               " cannot annotate physical type Time64");
  }
}

Result<std::shared_ptr<ArrowType>> MakeArrowTimestamp(const LogicalType& logical_type) {
  const auto& timestamp = checked_cast<const TimestampLogicalType&>(logical_type);
  const bool utc_normalized = timestamp.is_adjusted_to_utc();
  static const char* utc_timezone = "UTC";
  switch (timestamp.time_unit()) {
    case LogicalType::TimeUnit::MILLIS:
      return (utc_normalized ? ::arrow::timestamp(::arrow::TimeUnit::MILLI, utc_timezone)
                             : ::arrow::timestamp(::arrow::TimeUnit::MILLI));
    case LogicalType::TimeUnit::MICROS:
      return (utc_normalized ? ::arrow::timestamp(::arrow::TimeUnit::MICRO, utc_timezone)
                             : ::arrow::timestamp(::arrow::TimeUnit::MICRO));
    case LogicalType::TimeUnit::NANOS:
      return (utc_normalized ? ::arrow::timestamp(::arrow::TimeUnit::NANO, utc_timezone)
                             : ::arrow::timestamp(::arrow::TimeUnit::NANO));
    default:
      return Status::TypeError("Unrecognized time unit in timestamp logical_type: ",
                               logical_type.ToString());
  }
}

Result<std::shared_ptr<ArrowType>> FromByteArray(
    const LogicalType& logical_type, const ArrowReaderProperties& reader_properties,
    const std::shared_ptr<const ::arrow::KeyValueMetadata>& metadata) {
  auto binary_type = [&]() -> Result<std::shared_ptr<ArrowType>> {
    const auto configured_binary_type = reader_properties.binary_type();
    switch (configured_binary_type) {
      case ::arrow::Type::BINARY:
        return ::arrow::binary();
      case ::arrow::Type::LARGE_BINARY:
        return ::arrow::large_binary();
      case ::arrow::Type::BINARY_VIEW:
        return ::arrow::binary_view();
      default:
        return Status::TypeError("Invalid Arrow type for BYTE_ARRAY columns: ",
                                 ::arrow::internal::ToString(configured_binary_type));
    }
  };

  auto utf8_type = [&]() -> Result<std::shared_ptr<ArrowType>> {
    const auto configured_binary_type = reader_properties.binary_type();
    switch (configured_binary_type) {
      case ::arrow::Type::BINARY:
        return ::arrow::utf8();
      case ::arrow::Type::LARGE_BINARY:
        return ::arrow::large_utf8();
      case ::arrow::Type::BINARY_VIEW:
        return ::arrow::utf8_view();
      default:
        return Status::TypeError("Invalid Arrow type for BYTE_ARRAY columns: ",
                                 ::arrow::internal::ToString(configured_binary_type));
    }
  };

  switch (logical_type.type()) {
    case LogicalType::Type::STRING:
      return utf8_type();
    case LogicalType::Type::DECIMAL:
      return MakeArrowDecimal(logical_type);
    case LogicalType::Type::NONE:
    case LogicalType::Type::ENUM:
    case LogicalType::Type::BSON:
      return binary_type();
    case LogicalType::Type::JSON: {
      if (reader_properties.get_arrow_extensions_enabled()) {
        return utf8_type().Map(::arrow::extension::json);
      }
      // When the original Arrow schema isn't stored and Arrow extensions are disabled,
      // LogicalType::JSON is read as utf8().
      return utf8_type();
    }
    case LogicalType::Type::GEOMETRY:
    case LogicalType::Type::GEOGRAPHY: {
      ARROW_ASSIGN_OR_RAISE(auto storage_type, binary_type());
      if (reader_properties.get_arrow_extensions_enabled()) {
        // Attempt creating a GeoArrow extension type (or return the default
        // binary_type if types are not registered)
        return GeoArrowTypeFromLogicalType(logical_type, metadata, storage_type);
      }

      // When the original Arrow schema isn't stored, Arrow extensions are disabled, or
      // the geoarrow.wkb extension type isn't registered, LogicalType::GEOMETRY and
      // LogicalType::GEOGRAPHY are read as the default binary_type.
      return storage_type;
    }
    default:
      return Status::NotImplemented("Unhandled logical logical_type ",
                                    logical_type.ToString(), " for binary array");
  }
}

Result<std::shared_ptr<ArrowType>> FromFLBA(
    const LogicalType& logical_type, int32_t physical_length,
    const ArrowReaderProperties& reader_properties) {
  switch (logical_type.type()) {
    case LogicalType::Type::DECIMAL:
      return MakeArrowDecimal(logical_type);
    case LogicalType::Type::FLOAT16:
      return ::arrow::float16();
    case LogicalType::Type::NONE:
    case LogicalType::Type::INTERVAL:
      return ::arrow::fixed_size_binary(physical_length);
    case LogicalType::Type::UUID:
      if (physical_length == 16 && reader_properties.get_arrow_extensions_enabled()) {
        return ::arrow::extension::uuid();
      }

      return ::arrow::fixed_size_binary(physical_length);
    default:
      return Status::NotImplemented("Unhandled logical_type ", logical_type.ToString(),
                                    " for fixed-length binary array");
  }
}

}  // namespace

::arrow::Result<std::shared_ptr<ArrowType>> FromInt32(const LogicalType& logical_type) {
  switch (logical_type.type()) {
    case LogicalType::Type::INT:
      return MakeArrowInt(logical_type);
    case LogicalType::Type::DATE:
      return ::arrow::date32();
    case LogicalType::Type::TIME:
      return MakeArrowTime32(logical_type);
    case LogicalType::Type::DECIMAL:
      return MakeArrowDecimal(logical_type);
    case LogicalType::Type::NONE:
      return ::arrow::int32();
    default:
      return Status::NotImplemented("Unhandled logical type ", logical_type.ToString(),
                                    " for INT32");
  }
}

Result<std::shared_ptr<ArrowType>> FromInt64(const LogicalType& logical_type) {
  switch (logical_type.type()) {
    case LogicalType::Type::INT:
      return MakeArrowInt64(logical_type);
    case LogicalType::Type::DECIMAL:
      return MakeArrowDecimal(logical_type);
    case LogicalType::Type::TIMESTAMP:
      return MakeArrowTimestamp(logical_type);
    case LogicalType::Type::TIME:
      return MakeArrowTime64(logical_type);
    case LogicalType::Type::NONE:
      return ::arrow::int64();
    default:
      return Status::NotImplemented("Unhandled logical type ", logical_type.ToString(),
                                    " for INT64");
  }
}

Result<std::shared_ptr<ArrowType>> GetArrowType(
    Type::type physical_type, const LogicalType& logical_type, int type_length,
    const ArrowReaderProperties& reader_properties,
    const std::shared_ptr<const ::arrow::KeyValueMetadata>& metadata) {
  if (logical_type.is_null()) {
    return ::arrow::null();
  }

  if (logical_type.is_invalid()) {
    return GetArrowType(physical_type, *NoLogicalType::Make(), type_length,
                        reader_properties);
  }

  switch (physical_type) {
    case ParquetType::BOOLEAN:
      return ::arrow::boolean();
    case ParquetType::INT32:
      return FromInt32(logical_type);
    case ParquetType::INT64:
      return FromInt64(logical_type);
    case ParquetType::INT96:
      return ::arrow::timestamp(reader_properties.coerce_int96_timestamp_unit());
    case ParquetType::FLOAT:
      return ::arrow::float32();
    case ParquetType::DOUBLE:
      return ::arrow::float64();
    case ParquetType::BYTE_ARRAY:
      return FromByteArray(logical_type, reader_properties, metadata);
    case ParquetType::FIXED_LEN_BYTE_ARRAY:
      return FromFLBA(logical_type, type_length, reader_properties);
    default: {
      // PARQUET-1565: This can occur if the file is corrupt
      return Status::IOError("Invalid physical column type: ",
                             TypeToString(physical_type));
    }
  }
}

Result<std::shared_ptr<ArrowType>> GetArrowType(
    const schema::PrimitiveNode& primitive,
    const ArrowReaderProperties& reader_properties,
    const std::shared_ptr<const ::arrow::KeyValueMetadata>& metadata) {
  return GetArrowType(primitive.physical_type(), *primitive.logical_type(),
                      primitive.type_length(), reader_properties, metadata);
}

}  // namespace parquet::arrow
