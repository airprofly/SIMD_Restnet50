/**
File Name: timer.h 
Author: airprofly
Date: 2025-04-01
Description: This file has realized the following functions:
            - get the current time stamp in milliseconds
 
*/


#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>

/**
 * @brief get the time stamp in milliseconds

 * @return the time stamp in milliseconds
 */
inline int getTime(){
    int timeStamp=std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();//get the current time stamp in milliseconds
    return timeStamp;
}

#endif