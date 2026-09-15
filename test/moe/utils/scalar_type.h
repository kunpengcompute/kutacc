/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <cstdint>
#include "check.h"

namespace utils {
enum class ScalarType {
    Unknown,
    Int8,
    UInt8,
    Int16,
    Float16,
    BFloat16,
    Int32,
    Float32,
    Int64,
};

inline int64_t elementSize(ScalarType dtype)
{
    switch (dtype) {
        case ScalarType::Int8:
        case ScalarType::UInt8:
            return 1;
        case ScalarType::Int16:
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return 2;
        case ScalarType::Int32:
        case ScalarType::Float32:
            return 4;
        case ScalarType::Int64:
            return 8;
        default:
            PARAMETER_CHECK(false, int(dtype));
            return 0;
    }
}

inline ScalarType convertStringToScalarType(std::string name)
{
    if (name == "I8") {
        return ScalarType::Int8;
    } else if (name == "U8") {
        return ScalarType::UInt8;
    } else if (name == "I16") {
        return ScalarType::Int16;
    } else if (name == "F16") {
        return ScalarType::Float16;
    } else if (name == "BF16") {
        return ScalarType::BFloat16;
    } else if (name == "I32") {
        return ScalarType::Int32;
    } else if (name == "F32") {
        return ScalarType::Float32;
    } else if (name == "I64") {
        return ScalarType::Int64;
    } else {
        PARAMETER_CHECK(false, name);
        return ScalarType::Unknown;
    }
}

inline std::string convertScalarTypeToString(ScalarType type)
{
    if (type == ScalarType::Unknown) {
        return "Unknown";
    } else if (type == ScalarType::Int8) {
        return "I8";
    } else if (type == ScalarType::UInt8) {
        return "U8";
    } else if (type == ScalarType::Int16) {
        return "I16";
    } else if (type == ScalarType::Float16) {
        return "F16";
    } else if (type == ScalarType::BFloat16) {
        return "BF16";
    } else if (type == ScalarType::Int32) {
        return "I32";
    } else if (type == ScalarType::Float32) {
        return "F32";
    } else if (type == ScalarType::Int64) {
        return "I64";
    } else {
        PARAMETER_CHECK(false, int(type));
        return "Unknown";
    }
}
}  // namespace utils
