#include<iostream>


#include"alignPtr.h"

/**
 * @brief check if the pointer is aligned

 * @param alignment the memory required to be alloacted

 * @return if the pointer is aligned,return true;
            otherwise return false
 */
bool isAligned(void *ptr, size_t alignment){
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief automatic aligned memory allocation

 * @param size the memory required to be alloacted
 * @param alignment the aligned memory number
 *
 * @return the pointer to the aligned memory
 */
void *aligned_alloc(size_t size, size_t alignment){
    //process the exception
    if(alignment<=0||size<=0){
        std::cerr<<"alignment or size must be greater than 0"<<std::endl;
    }
    //allocate the memory for different platforms
    #if defined(_WIN32)||defined(_WIN64)
        void* ptr=_aligned_malloc(size,alignment);
        if(ptr==nullptr){
            std::cerr<<"_aligned_malloc failed"<<std::endl;
        }
        return ptr;
    #else
        void* ptr=nullptr;
        if(posix_memalign(&ptr,alignment,size)!=0){//if allocation succeed,the return value is 0;
            std::cerr<<"posix_memalign failed"<<std::endl;
        }
        return ptr;
    #endif
}

/**
 * @brief free the allocated aligned memory

 * @param ptr the pointer to the aligned memory
 */

void aligned_free(void *ptr){
    if(ptr==nullptr){
        std::cerr<<"the pointer is nullptr,it can't be freed"<<std::endl;
        return;
    }
    #if defined(_WIN32)||defined(_WIN64)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}



