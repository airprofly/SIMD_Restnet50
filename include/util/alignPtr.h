/**
File Name: alignPtr.h 
Author: airprofly
Email: 3502196756@qq.com
Date: 2025-04-01
Description: This file has realized the following functions:
            - aligned memory allocation
            - aligned memory free
            - make the aligned memory unique_ptr
*/


#ifndef _ALIGNPTR_H_
#define _ALIGNPTR_H_

#include <stddef.h>
#include <memory>
#include <iostream>

/**
 * @brief check if the pointer is aligned

 * @param alignment the memory required to be alloacted

 * @return if the pointer is aligned,return true;
            otherwise return false
 */

bool isAligned(void *ptr, size_t alignment);

/**
 * @brief automatic aligned memory allocation

 * @param size the aligned memory size
 * @param alignment the memory required to be alloacted

 * @return the pointer to the aligned memory
 */

void *aligned_alloc(size_t size, size_t alignment);

/**
 * @brief free the allocated aligned memory

 * @param ptr the pointer to the aligned memory
 */

void aligned_free(void *ptr);

// custom deleter for aligned memory
struct AlignDeleter
{
    void operator()(void *ptr) const noexcept
    {
        if(ptr == nullptr)return;
        aligned_free(ptr);
        ptr = nullptr;
    }
};


template <typename T>
using aligned_unique_ptr = std::unique_ptr<T[], AlignDeleter>;
/**
 * @brief the template for allocating the aligned memory
 * @param T the type of the aligned memory
 * @param count the number of the stored_type data
 * @param alignment the aligned memory number
 * @return the unique_ptr to the aligned memory
 * @note the aligned memory will be freed automatically
 */
template <typename T>
aligned_unique_ptr<T> make_aligned_unique(size_t count,    // the number of the stored_type data
                                          size_t alignment // the aligned memory number
)
{
    // process the exception
    if (count <= 0 || alignment <= 0)
    {
        std::cerr << "size or alignment must be greater than 0" << std::endl;
        return nullptr;
    }
    if (alignment < sizeof(T))
    {
        std::cerr << "the alignment must be greater than sizeof(T)" << std::endl;
        return nullptr;
    }
    // allocate the memory
    void *ptr = aligned_alloc(count * sizeof(T), alignment);
    return std::move(std::unique_ptr<T[], AlignDeleter>(static_cast<T *>(ptr)));
}

#endif