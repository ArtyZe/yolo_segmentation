/*
 *  Copyright 2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include "memory_resource.h"

namespace thrust
{
namespace mr
{

template<typename Pointer = void *>
class polymorphic_adaptor_resource THRUST_FINAL : public memory_resource<Pointer>
{
public:
    polymorphic_adaptor_resource(memory_resource<Pointer> * t) : upstream_resource(t)
    {
    }

    virtual void * do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) THRUST_OVERRIDE
    {
        return upstream_resource->allocate(bytes, alignment);
    }

    virtual void do_deallocate(void * p, std::size_t bytes, std::size_t alignment) THRUST_OVERRIDE
    {
        return upstream_resource->deallocate(p, bytes, alignment);
    }

    __host__ __device__
    virtual bool do_is_equal(const memory_resource<Pointer> & other) const THRUST_NOEXCEPT THRUST_OVERRIDE
    {
        return upstream_resource->is_equal(other);
    }

private:
    memory_resource<Pointer> * upstream_resource;
};

} // end mr
} // end thrust

